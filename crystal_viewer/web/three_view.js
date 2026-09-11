import * as THREE from "/vendor/three/three.module.js";
import { TrackballControls } from "/vendor/three/addons/controls/TrackballControls.js";
import {
  applyBoundaryContext,
  evaluatePath,
  pathBreakpoints,
  pathAppliesToDisplayInstance,
  sequentialSegmentBoundaries,
} from "/static/animation_path.js";
import { SYMMETRY_ELEMENT_COLORS } from "/static/colors.js";


const CAMERA_FOV = 42;
const ORTHOGRAPHIC_HEIGHT = 10;

function viewerConstants() {
  // Injected by crystal_viewer/web/browser_ui.py so Python owns the value.
  // The literals below are a last resort for contexts without the page (the
  // JS syntax check and the node test harness), not a second definition.
  try {
    return JSON.parse(document.getElementById("viewer-constants").textContent);
  } catch {
    return {};
  }
}
const STATIONARY_ANIMATION_SECONDS = viewerConstants().stationary_animation_seconds ?? 0.4;
const GIF_END_HOLD_MS = 1000;
const CURVED_PATH_TYPES = new Set(["rotation", "screw", "rotoinversion", "rotoreflection"]);
const CRYSTAL_AXIS_COLORS = [0xef4444, 0x22c55e, 0x3b82f6];
const ZERO_SCALE = new THREE.Vector3(0, 0, 0);


function pathHasCurvedMotion(path) {
  if (!path) return false;
  if (path.type === "sequential") return (path.segments || []).some(pathHasCurvedMotion);
  return CURVED_PATH_TYPES.has(path.type);
}

export class StaticStructureView {
  constructor(container) {
    this.container = container;
    this.status = container.querySelector("[data-three-status]");
    this.legend = container.querySelector("[data-atom-legend]");
    this.canvas = container.querySelector("canvas");
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf4f6f3);
    this.content = new THREE.Group();
    this.scene.add(this.content);

    this.perspectiveCamera = new THREE.PerspectiveCamera(CAMERA_FOV, 1, 0.01, 10000);
    this.orthographicCamera = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.01, 10000);
    this.activeCamera = this.perspectiveCamera;
    this.renderer = new THREE.WebGLRenderer({canvas: this.canvas, antialias: true, preserveDrawingBuffer: true});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Both views live for the whole session; only the visible one draws.
    this.active = true;
    this.frameHandle = null;
    this.controls = null;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.pointerDown = null;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x263342, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
    keyLight.position.set(5, -4, 7);
    this.scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0x8ec5ff, 1.2);
    fillLight.position.set(-4, 3, -2);
    this.scene.add(fillLight);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.lastStateSignature = "";
    this.state = {};
    this.atomInstances = new Map();
    this.instanceMeshes = new Set();
    this.animationPaths = new Map();
    this.boundaryContext = {mode: "continuous"};
    this.animationOperationIndex = null;
    this.customAnimationId = null;
    this.customRepresentativePath = null;
    this.customSegmentElements = [];
    this.customSegmentBoundaries = [];
    this.customSegmentIndex = null;
    this.animationBreakpoints = [0, 1];
    this.symmetryOperationIndex = null;
    this.operationViewDirection = null;
    this.operationFocusPoint = null;
    this.symmetryObjects = [];
    this.targetMarkerObjects = [];
    this.startMarkerObjects = [];
    this.trajectoryObjects = [];
    this.selectionMarkers = new Map();
    this.selectionSignature = null;
    // Puzzle-owned atom rings (source / guess / target), independent of the
    // analysis selection. Set via setPickMarkers; an onAtomPick hook, when
    // present, routes clicks to the puzzle instead of the analysis selection.
    this.pickMarkers = new Map();
    this.onAtomPick = null;
    this.disableAtomSelection = false;
    // Optional source-atom filter used by the mapping puzzle preview. Paths remain
    // loaded for the full operation, but atoms outside the set stay at rest.
    this.animationSourceAtoms = null;
    this.animationProgress = 0;
    this.animationStartedAt = null;
    this.playing = false;
    this.maximumTravelDistance = 0;
    this.baseAnimationDurationSeconds = STATIONARY_ANIMATION_SECONDS;
    this.baseStatus = "3D view";
    this.lastProgressBucket = -1;
    this.lastPublishedProgressBucket = -1;
    this.pathGeneration = 0;
    this.syncQueue = Promise.resolve();
    this.serverPlaying = false;
    this.recording = false;
    this.backgroundMode = null;
    this.renderDataQuery = ""; // owner may set e.g. "?boundary_images=1" (puzzle)
    this.animationPathQuery = ""; // owner may set e.g. "&display_mode=source" (puzzle)
    this.symmetryElementQuery = ""; // owner may set e.g. "&display_mode=source" (puzzle)
    this.showAnimationTargets = false; // owner may enable static target markers (puzzle)
    this.showAnimationTargetCopies = true;
    this.showTrajectories = false;
    this.hiddenElements = new Set();
    this.legendItems = [];
    this.tempMatrix = new THREE.Matrix4();
    this.tempPosition = new THREE.Vector3();
    this.tempScale = new THREE.Vector3();
    this.tempQuaternion = new THREE.Quaternion();
    this.animate = this.animate.bind(this);
    this.handlePointerDown = this.handlePointerDown.bind(this);
    this.handlePointerUp = this.handlePointerUp.bind(this);
    this.canvas.addEventListener("pointerdown", this.handlePointerDown);
    this.canvas.addEventListener("pointerup", this.handlePointerUp);
    window.addEventListener("symmetry-animation-progress", event => {
      this.setAnimationProgress(event.detail?.progress);
    });
    window.addEventListener("symmetry-save-png", () => this.savePng());
    window.addEventListener("symmetry-save-gif", () => {
      this.syncQueue.then(() => this.recordGif());
    });
    this.requestFrame();
  }

  async renderPayload() {
    this.setStatus("Loading 3D view…");
    // Optional query (e.g. the puzzle's "?boundary_images=1") set by the owner.
    const response = await fetch(`/api/render_data${this.renderDataQuery || ""}`);
    if (!response.ok) {
      throw new Error(await response.text() || `${response.status} ${response.statusText}`);
    }
    const payload = await response.json();
    if (payload.coordinate_space !== "cartesian") {
      throw new Error(`Unsupported coordinate space: ${payload.coordinate_space}`);
    }
    return payload;
  }

  async refresh(preserveCamera = false) {
    const cameraState = preserveCamera && this.controls ? {
      position: this.activeCamera.position.clone(),
      up: this.activeCamera.up.clone(),
      quaternion: this.activeCamera.quaternion.clone(),
      target: this.controls.target.clone(),
    } : null;
    const payload = await this.renderPayload();
    this.setData(payload);
    if (cameraState) {
      this.activeCamera.position.copy(cameraState.position);
      this.activeCamera.up.copy(cameraState.up);
      this.activeCamera.quaternion.copy(cameraState.quaternion);
      this.controls.target.copy(cameraState.target);
      this.controls.update();
      this.render();
    }
  }

  async refreshAtomColors() {
    const payload = await this.renderPayload();
    const styles = new Map((payload.atom_styles || []).map(style => [Number(style.index), style]));
    for (const mesh of this.instanceMeshes) {
      for (const instance of mesh.userData.atomInstances || []) {
        const style = styles.get(instance.sourceAtom);
        if (!style?.color) continue;
        instance.baseColor.set(style.color);
        mesh.setColorAt(instance.instanceIndex, instance.baseColor);
      }
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
    this.setLegendItems(payload.display_atoms || [], payload.atom_styles || []);
    this.render();
  }

  setData(payload) {
    this.clearContent();
    this.hiddenElements.clear();
    this.isMolecule = payload.source_kind === "molecule";
    const renderData = payload.render_data || {};
    this.renderData = renderData;
    const atoms = payload.display_atoms || renderData.atoms || [];
    const styles = new Map((payload.atom_styles || []).map(style => [Number(style.index), style]));
    this.setLegendItems(atoms, payload.atom_styles || []);
    this.addAtoms(atoms, styles);
    if (payload.source_kind === "crystal" && payload.display_unit_cell) {
      this.addUnitCell(payload.display_unit_cell, renderData.unit_cell);
    }
    this.fitCamera(renderData);
    this.animationOperationIndex = null;
    this.symmetryOperationIndex = null;
    const kind = payload.source_kind === "molecule" ? "molecule" : "crystal";
    this.baseStatus = `3D view: ${atoms.length} atoms (${kind})`;
    this.setStatus(`${this.baseStatus}, animation ready`);
    this.render();
  }

  addAtoms(atoms, styles) {
    this.atomInstances.clear();
    const groups = new Map();
    for (const atom of atoms) {
      if (!Array.isArray(atom.cart) || atom.cart.length !== 3) continue;
      const sourceAtom = Number(atom.source_atom ?? atom.index);
      const style = styles.get(sourceAtom) || {color: "#9aa5b1", radius: 0.35};
      const radius = Math.max(Number(style.radius) || 0.35, 0.02);
      const key = `${style.color}|${radius.toFixed(8)}`;
      if (!groups.has(key)) groups.set(key, {style, radius, atoms: []});
      groups.get(key).atoms.push(atom);
    }
    this.maxAtomRadius = groups.size
      ? Math.max(...[...groups.values()].map(group => group.radius))
      : 0.35;

    const geometry = new THREE.SphereGeometry(1, 28, 18);
    const matrix = new THREE.Matrix4();
    for (const group of groups.values()) {
      const material = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.34,
        metalness: 0.04,
      });
      const mesh = new THREE.InstancedMesh(geometry, material, group.atoms.length);
      mesh.frustumCulled = false;
      mesh.userData.atomInstances = [];
      this.instanceMeshes.add(mesh);
      group.atoms.forEach((atom, instanceIndex) => {
        matrix.compose(
          new THREE.Vector3(...atom.cart),
          new THREE.Quaternion(),
          new THREE.Vector3(group.radius, group.radius, group.radius),
        );
        mesh.setMatrixAt(instanceIndex, matrix);
        const instanceId = Number(atom.instance_id ?? atom.index);
        const instance = {
          mesh,
          instanceIndex,
          sourceAtom: Number(atom.source_atom ?? atom.index),
          element: atom.element || "",
          isPrimaryImage: atom.is_primary_image !== false,
          radius: group.radius,
          start: [...atom.cart],
          current: [...atom.cart],
          baseColor: new THREE.Color(group.style.color),
        };
        mesh.setColorAt(instanceIndex, instance.baseColor);
        mesh.userData.atomInstances[instanceIndex] = instance;
        this.atomInstances.set(instanceId, instance);
      });
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      this.content.add(mesh);
    }
    this.updateAtomSelectionHighlight();
  }

  addUnitCell(unitCell, sourceUnitCell = null) {
    const vertices = unitCell.vertices_cart || [];
    const lattice = sourceUnitCell?.lattice || [];
    const directions = Array.isArray(lattice)
      ? lattice.map(vector => new THREE.Vector3(...vector).normalize())
      : [];
    const positionsByAxis = [[], [], [], []];
    for (const edge of unitCell.edges || []) {
      const start = vertices[edge[0]];
      const end = vertices[edge[1]];
      if (!start || !end) continue;
      const edgeDirection = new THREE.Vector3(
        end[0] - start[0],
        end[1] - start[1],
        end[2] - start[2],
      ).normalize();
      let axis = 3;
      let best = -1;
      for (let index = 0; index < directions.length; index += 1) {
        const score = Math.abs(edgeDirection.dot(directions[index]));
        if (score > best) {
          best = score;
          axis = index;
        }
      }
      positionsByAxis[Math.min(axis, 3)].push(...start, ...end);
    }
    for (let axis = 0; axis < positionsByAxis.length; axis += 1) {
      const positions = positionsByAxis[axis];
      if (!positions.length) continue;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      const color = axis < 3 ? CRYSTAL_AXIS_COLORS[axis] : 0xa8b5c5;
      const material = new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.86});
      this.content.add(new THREE.LineSegments(geometry, material));
    }
  }

  clearContent() {
    this.symmetryObjects = [];
    this.clearTargetMarkers();
    this.startMarkerObjects = [];
    this.trajectoryObjects = [];
    this.selectionMarkers.clear();
    this.selectionSignature = null;
    this.pickMarkers.clear();
    this.atomInstances.clear();
    this.instanceMeshes.clear();
    for (const child of [...this.content.children]) {
      this.content.remove(child);
      child.traverse(object => {
        object.geometry?.dispose();
        if (Array.isArray(object.material)) object.material.forEach(material => material.dispose());
        else object.material?.dispose();
      });
    }
  }

  fitCamera(renderData) {
    const box = new THREE.Box3().setFromObject(this.content);
    if (box.isEmpty()) {
      const minimum = renderData.bounds_min || [-1, -1, -1];
      const maximum = renderData.bounds_max || [1, 1, 1];
      box.set(new THREE.Vector3(...minimum), new THREE.Vector3(...maximum));
    }
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const span = Math.max(size.x, size.y, size.z, 1);
    const distance = span / (2 * Math.tan(THREE.MathUtils.degToRad(CAMERA_FOV / 2))) * 1.45;
    const position = center.clone().add(new THREE.Vector3(1, -1.35, 0.9).normalize().multiplyScalar(distance));
    for (const camera of [this.perspectiveCamera, this.orthographicCamera]) {
      camera.position.copy(position);
      camera.up.set(0, 0, 1);
      camera.lookAt(center);
      camera.near = Math.max(distance / 1000, 0.001);
      camera.far = Math.max(distance * 20, 100);
      camera.updateProjectionMatrix();
    }
    this.orthographicSpan = span * 1.45;
    this.createControls(center);
    // This is captured before symmetry elements are added, so a reset returns
    // to the same framing used when the structure was first displayed.
    this.defaultCameraState = {
      position: this.activeCamera.position.clone(),
      up: this.activeCamera.up.clone(),
      target: this.controls.target.clone(),
    };
    this.resize();
  }

  createControls(target) {
    this.controls?.dispose();
    this.controls = new TrackballControls(this.activeCamera, this.canvas);
    this.controls.rotateSpeed = 1.15;
    this.controls.zoomSpeed = 1.05;
    this.controls.panSpeed = 0.65;
    this.controls.staticMoving = false;
    this.controls.dynamicDampingFactor = 0.14;
    this.controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    this.controls.mouseButtons.MIDDLE = THREE.MOUSE.DOLLY;
    this.controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    this.controls.target.copy(target);
    this.controls.addEventListener("change", () => this.render());
    this.controls.handleResize();
    this.controls.update();
  }

  setProjection(mode) {
    const nextCamera = mode === "orthographic" ? this.orthographicCamera : this.perspectiveCamera;
    if (nextCamera === this.activeCamera) return;
    nextCamera.position.copy(this.activeCamera.position);
    nextCamera.quaternion.copy(this.activeCamera.quaternion);
    const target = this.controls?.target.clone() || new THREE.Vector3();
    this.activeCamera = nextCamera;
    this.createControls(target);
    this.resize();
  }

  handlePointerDown(event) {
    if (event.button !== 0) return;
    this.pointerDown = {
      x: event.clientX,
      y: event.clientY,
      time: performance.now(),
    };
  }

  handlePointerUp(event) {
    if (event.button !== 0 || !this.pointerDown) return;
    const movement = Math.hypot(event.clientX - this.pointerDown.x, event.clientY - this.pointerDown.y);
    const elapsed = performance.now() - this.pointerDown.time;
    this.pointerDown = null;
    if (movement > 5 || elapsed > 500) return;
    const hit = this.pickAtomInstance(event);
    if (this.onAtomPick) {
      // Puzzle mode owns clicks (e.g. the "where does this atom go?" quiz); do not
      // touch the shared analysis selection / server state.
      this.onAtomPick(hit ? hit.sourceAtom : null);
      return;
    }
    if (this.disableAtomSelection) return;
    if (!hit && !String(this.state.scope).startsWith("selected")) return;
    this.selectAtom(hit ? hit.sourceAtom : null, event);
  }

  pickAtomInstance(event) {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.activeCamera);
    const intersections = this.raycaster.intersectObjects([...this.instanceMeshes], false);
    for (const intersection of intersections) {
      const instanceIndex = intersection.instanceId;
      if (!Number.isInteger(instanceIndex)) continue;
      const instance = intersection.object.userData.atomInstances?.[instanceIndex];
      if (instance && !this.isAtomVisible(instance)) continue;
      if (instance) return instance;
    }
    return null;
  }

  async selectAtom(sourceAtom, event) {
    const current = new Set((this.state.selected_atoms || []).map(Number));
    let selected;
    if (sourceAtom === null) {
      selected = [];
    } else if (event.shiftKey || event.ctrlKey || event.metaKey) {
      if (current.has(sourceAtom)) current.delete(sourceAtom);
      else current.add(sourceAtom);
      selected = [...current].sort((a, b) => a - b);
    } else {
      selected = current.size === 1 && current.has(sourceAtom) ? [] : [sourceAtom];
    }
    const update = {
      selected_atoms: selected,
      scope: selected.length ? "selected_displayed" : "displayed",
      playing: false,
      reset: true,
    };
    try {
      if (typeof window.symmetryPostState === "function") {
        await window.symmetryPostState(update);
        return;
      }
      const response = await fetch("/api/state", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(update),
      });
      if (!response.ok) throw new Error(await response.text() || `${response.status} ${response.statusText}`);
      const state = await response.json();
      window.dispatchEvent(new CustomEvent("symmetry-state-update", {
        detail: {state: {...state}, update: {...update}},
      }));
    } catch (error) {
      this.setStatus(`Selection error: ${error.message}`);
    }
  }

  updateAtomSelectionHighlight() {
    const selected = String(this.state.scope).startsWith("selected")
      ? new Set((this.state.selected_atoms || []).map(Number))
      : new Set();
    const signature = JSON.stringify([...selected].sort((a, b) => a - b));
    if (signature === this.selectionSignature) return;
    this.selectionSignature = signature;
    this.updateSelectionMarkers(selected);
    this.render();
  }

  updateSelectionMarkers(selected) {
    for (const marker of this.selectionMarkers.values()) {
      this.content.remove(marker);
      marker.geometry.dispose();
      marker.material.dispose();
    }
    this.selectionMarkers.clear();
    for (const [instanceId, instance] of this.atomInstances) {
      if (!selected.has(instance.sourceAtom)) continue;
      const outerRadius = instance.radius * 1.36;
      const geometry = new THREE.RingGeometry(outerRadius * 0.94, outerRadius, 48);
      const material = new THREE.MeshBasicMaterial({
        color: 0xffe45c,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.58,
        depthTest: true,
        depthWrite: false,
      });
      const marker = new THREE.Mesh(geometry, material);
      marker.position.fromArray(instance.current);
      marker.renderOrder = 10;
      this.selectionMarkers.set(instanceId, marker);
      this.content.add(marker);
    }
  }

  clearPickMarkers() {
    for (const marker of this.pickMarkers.values()) {
      this.content.remove(marker);
      marker.geometry.dispose();
      marker.material.dispose();
    }
    this.pickMarkers.clear();
    this.container.dataset.pickMarkerCount = "0";
    this.render();
  }

  setAnimationSourceAtoms(sourceAtoms = null) {
    this.animationSourceAtoms = sourceAtoms == null
      ? null
      : new Set([...sourceAtoms].map((index) => Number(index)));
  }

  animationPathForInstance(instance) {
    if (this.animationSourceAtoms && !this.animationSourceAtoms.has(instance.sourceAtom)) {
      return null;
    }
    return this.animationPaths.get(instance.sourceAtom);
  }

  _makePickRing(color, radius) {
    const outerRadius = radius * 1.5;
    const geometry = new THREE.RingGeometry(outerRadius * 0.9, outerRadius, 48);
    const material = new THREE.MeshBasicMaterial({
      color,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
      depthTest: true,
      depthWrite: false,
    });
    const marker = new THREE.Mesh(geometry, material);
    marker.quaternion.copy(this.activeCamera.quaternion);
    marker.renderOrder = 11;
    return marker;
  }

  // entries: each is either
  //   {atom: <sourceAtom index>, color} — a ring on that atom (and every displayed
  //     instance sharing the source atom); follows the atom's animated position, so
  //     a highlighted atom stays ringed as the operation plays, or
  //   {position: [x,y,z], color, radius?} — a fixed ring at a world position (e.g.
  //     the destination an atom will land on), which does NOT move with the animation.
  setPickMarkers(entries) {
    this.clearPickMarkers();
    let staticKey = -1;
    for (const entry of entries || []) {
      if (entry.position) {
        const marker = this._makePickRing(entry.color, entry.radius || 0.5);
        marker.position.fromArray(entry.position);
        // Negative keys are absent from atomInstances, so setAtomPosition never
        // repositions them — the ring stays put while the animation plays.
        this.pickMarkers.set(staticKey--, marker);
        this.content.add(marker);
        continue;
      }
      for (const [instanceId, instance] of this.atomInstances) {
        if (instance.sourceAtom !== Number(entry.atom)) continue;
        const marker = this._makePickRing(entry.color, instance.radius);
        marker.position.fromArray(instance.current);
        marker.visible = this.isAtomVisible(instance);
        this.pickMarkers.set(instanceId, marker);
        this.content.add(marker);
      }
    }
    this.container.dataset.pickMarkerCount = String(this.pickMarkers.size);
    this.render();
  }

  syncState(state, update = {}) {
    this.syncQueue = this.syncQueue
      .catch(() => {})
      .then(() => this.applyState(state, update));
    return this.syncQueue;
  }

  async applyState(state, update = {}) {
    const previousOperation = this.state.operation_index;
    const displayLayoutChanged = this.state.operation_index !== undefined && (
      state.display_mode !== this.state.display_mode
      || state.cell_origin_mode !== this.state.cell_origin_mode
      || state.include_boundary_images !== this.state.include_boundary_images
    );
    const atomVisibilityChanged = ["element_hidden", "atom_hidden"]
      .some(key => Object.prototype.hasOwnProperty.call(update, key));
    const atomColorsChanged = ["element_colors", "atom_colors"]
      .some(key => Object.prototype.hasOwnProperty.call(update, key));
    const trajectorySettingChanged = Object.prototype.hasOwnProperty.call(update, "show_trajectories");
    this.state = {...state};
    this.setBackgroundMode(state.background_mode);
    this.setLegendVisible(state.legend_visible);
    if (displayLayoutChanged || atomVisibilityChanged) {
      await this.refresh(atomVisibilityChanged && !displayLayoutChanged);
      if (state.active_mode === "custom") {
        this.loadCustomSymmetryElements(state.custom_op_result);
        this.customSegmentIndex = null;
        this.updateCustomSequenceElements();
      }
    } else if (atomColorsChanged) {
      await this.refreshAtomColors();
    }
    this.updateAtomSelectionHighlight();
    this.setProjection(state.projection_mode);
    const operationChanged = Number(state.operation_index) !== Number(previousOperation);
    const customRequest = state.custom_op_animate || null;
    const customAnimationChanged = state.active_mode === "custom"
      && customRequest?.animate_id !== this.customAnimationId;
    // A Custom panel with no checked sequence has no animation to show.  Drop
    // the preceding Standard operation immediately so its trajectory, markers,
    // and symmetry elements cannot remain visible behind the unchecked UI.
    if (state.active_mode === "custom" && !customRequest && this.animationOperationIndex !== null) {
      this.pathGeneration += 1;
      this.animationPaths = new Map();
      this.animationOperationIndex = null;
      this.clearSymmetryElements();
      this.clearTargetMarkers();
      this.clearTrajectoryLines();
      this.clearStartMarkers();
      this.animationBreakpoints = [0, 1];
      window.dispatchEvent(new CustomEvent("symmetry-animation-breakpoints", {
        detail: {breakpoints: this.animationBreakpoints},
      }));
    }
    const pathOptionsChanged = operationChanged || [
      "scope",
      "selected_atoms",
      "improper_mode",
      "display_mode",
      "cell_origin_mode",
      "include_boundary_images",
      "animation_boundary_mode",
      "structure_reload",
    ].some(key => Object.prototype.hasOwnProperty.call(update, key));
    const customPathOptionsChanged = state.active_mode === "custom"
      && customRequest
      && (customAnimationChanged || pathOptionsChanged);
    let pathsUpdated = false;
    if (customPathOptionsChanged) {
      const generation = ++this.pathGeneration;
      await this.loadCustomAnimationPaths(generation);
      if (generation !== this.pathGeneration) return;
      this.loadCustomSymmetryElements(state.custom_op_result);
      this.resetAnimation();
      this.updateCustomSequenceElements();
      pathsUpdated = true;
    } else if (state.active_mode !== "custom" && (pathOptionsChanged || this.animationOperationIndex === null)) {
      const generation = ++this.pathGeneration;
      await Promise.all([
        this.loadAnimationPaths(Number(state.operation_index), generation),
        this.loadSymmetryElements(Number(state.operation_index), generation),
      ]);
      if (generation !== this.pathGeneration) return;
      this.resetAnimation();
      pathsUpdated = true;
    }
    if (pathsUpdated || atomVisibilityChanged || trajectorySettingChanged) this.updateTrajectoryLines();
    if (update.reset) {
      this.resetAnimation();
    }
    if (Object.prototype.hasOwnProperty.call(update, "camera_request_id")) {
      this.rotateCamera(update.camera_direction, update.camera_angle);
    }
    if (Object.prototype.hasOwnProperty.call(update, "view_request_id")) {
      this.viewAlongCurrentOperation();
    }
    if (Object.prototype.hasOwnProperty.call(update, "reset_view_request_id")) {
      this.resetCamera();
    }
    if (Object.prototype.hasOwnProperty.call(update, "view_center_request_id")) {
      this.setCameraCenter(state.view_center_cart);
    }
    if (Object.prototype.hasOwnProperty.call(update, "view_direction_request_id")) {
      this.viewAlongCartesianDirection(state.view_direction_cart, state.indexed_view_center_cart);
    }
    if (Object.prototype.hasOwnProperty.call(update, "view_plane_request_id")) {
      this.viewAlongCartesianDirection(state.view_plane_normal_cart, state.indexed_view_center_cart);
    }
    if (this.recording) return;
    const nextServerPlaying = Boolean(state.playing);
    const starting = nextServerPlaying && !this.serverPlaying;
    if (starting && this.animationProgress >= 1) {
      this.resetAnimation();
    }
    if (starting) this.showStartMarkers();
    this.serverPlaying = nextServerPlaying;
    this.playing = nextServerPlaying && this.animationProgress < 1;
    if (this.playing) {
      this.animationStartedAt = null;
    } else {
      this.animationStartedAt = null;
    }
  }

  async loadAnimationPaths(operationIndex, generation, scope = null) {
    if (!Number.isInteger(operationIndex)) return;
    // An optional scope override lets a caller (the puzzle) force "displayed" so
    // every atom animates regardless of the shared analysis-mode selection.
    const scopeParam = scope ? `&scope=${encodeURIComponent(scope)}` : "";
    const response = await fetch(`/api/animation_path?operation_index=${operationIndex}${scopeParam}${this.animationPathQuery || ""}`);
    if (!response.ok) {
      throw new Error(await response.text() || `${response.status} ${response.statusText}`);
    }
    const payload = await response.json();
    if (generation !== this.pathGeneration) return;
    this.animationPaths = new Map(
      (payload.paths || []).map(item => [Number(item.source_atom), item.path]),
    );
    const representativePath = this.animationPaths.values().next().value;
    const breakpoints = representativePath ? pathBreakpoints(representativePath) : [0, 1];
    this.animationBreakpoints = breakpoints;
    window.dispatchEvent(new CustomEvent("symmetry-animation-breakpoints", {
      detail: {breakpoints},
    }));
    this.boundaryContext = payload.boundary || {mode: "continuous"};
    this.animationOperationIndex = operationIndex;
    this.maximumTravelDistance = Math.max(Number(payload.maximum_travel_distance) || 0, 0);
    this.baseAnimationDurationSeconds = Math.max(
      Number(payload.animation_duration_seconds) || STATIONARY_ANIMATION_SECONDS,
      0.01,
    );
    this.customAnimationId = null;
    this.customRepresentativePath = null;
    this.customSegmentElements = [];
    this.customSegmentBoundaries = [];
    this.customSegmentIndex = null;
    this.updateAnimationTargetMarkers();
    this.updateTrajectoryLines();
  }

  async loadCustomAnimationPaths(generation) {
    const response = await fetch("/api/custom_animation_path");
    if (!response.ok) throw new Error(await response.text() || `${response.status} ${response.statusText}`);
    const payload = await response.json();
    if (generation !== this.pathGeneration) return;
    this.animationPaths = new Map(
      (payload.paths || []).map(item => [Number(item.source_atom), item.path]),
    );
    const representativePath = this.animationPaths.values().next().value;
    this.animationBreakpoints = representativePath ? pathBreakpoints(representativePath) : [0, 1];
    window.dispatchEvent(new CustomEvent("symmetry-animation-breakpoints", {
      detail: {breakpoints: this.animationBreakpoints},
    }));
    this.boundaryContext = payload.boundary || {mode: "continuous"};
    this.maximumTravelDistance = Math.max(Number(payload.maximum_travel_distance) || 0, 0);
    this.baseAnimationDurationSeconds = Math.max(
      Number(payload.animation_duration_seconds) || STATIONARY_ANIMATION_SECONDS, 0.01,
    );
    this.customAnimationId = payload.animate_id;
    this.customRepresentativePath = representativePath || null;
    this.customSegmentElements = representativePath?.segment_elements || [];
    this.customSegmentBoundaries = representativePath ? sequentialSegmentBoundaries(representativePath) : [];
    this.customSegmentIndex = null;
    this.animationOperationIndex = null;
  }

  loadCustomSymmetryElements(result) {
    this.clearSymmetryElements();
    const elements = result?.elements || {};
    const span = this.sceneSpan();
    for (const axis of elements.axes || []) this.addAxis(axis, span);
    for (const plane of elements.planes || []) this.addPlane(plane, span);
    for (const center of elements.centers || []) this.addCenter(center, span);
    if ((elements.planes || []).length && Array.isArray(elements.glide_translation_cart)) {
      this.addGlideArrow(elements.planes[0], elements.glide_translation_cart, span);
    }
    this.operationViewDirection = Array.isArray(result?.view_direction_cart)
      ? [...result.view_direction_cart] : null;
    this.operationFocusPoint = null;
    this.render();
  }

  updateCustomSequenceElements() {
    if (!this.customSegmentElements.length || !this.customRepresentativePath) return;
    const index = this.customSegmentBoundaries.findIndex(
      boundary => this.animationProgress <= boundary + 1e-12,
    );
    if (index === null || index === this.customSegmentIndex) return;
    this.customSegmentIndex = index;
    const elements = this.customSegmentElements[index] || {};
    this.clearSymmetryElements();
    const span = this.sceneSpan();
    for (const axis of elements.axes || []) this.addAxis(axis, span);
    for (const plane of elements.planes || []) this.addPlane(plane, span);
    for (const center of elements.centers || []) this.addCenter(center, span);
    if ((elements.planes || []).length && Array.isArray(elements.glide_translation_cart)) {
      this.addGlideArrow(elements.planes[0], elements.glide_translation_cart, span);
    }
    this.render();
  }

  async loadSymmetryElements(operationIndex, generation) {
    if (!Number.isInteger(operationIndex)) return;
    const response = await fetch(`/api/symmetry_elements?operation_index=${operationIndex}${this.symmetryElementQuery || ""}`);
    if (!response.ok) {
      throw new Error(await response.text() || `${response.status} ${response.statusText}`);
    }
    const payload = await response.json();
    if (generation !== this.pathGeneration) return;
    this.clearSymmetryElements();
    const span = this.sceneSpan();
    for (const axis of payload.axes || []) {
      this.addAxis(axis, span);
      this.addRotationArc(axis, span);
    }
    for (const plane of payload.planes || []) this.addPlane(plane, span);
    for (const center of payload.centers || []) this.addCenter(center, span);
    if ((payload.planes || []).length && Array.isArray(payload.glide_translation_cart)) {
      this.addGlideArrow(payload.planes[0], payload.glide_translation_cart, span);
    }
    this.symmetryOperationIndex = operationIndex;
    this.operationViewDirection = Array.isArray(payload.view_direction_cart)
      ? [...payload.view_direction_cart]
      : null;
    this.operationFocusPoint = Array.isArray(payload.focus_point_cart)
      ? [...payload.focus_point_cart]
      : null;
    this.render();
  }

  rotateCamera(direction, angleDegrees) {
    const angle = THREE.MathUtils.degToRad(Math.max(0, Math.min(Number(angleDegrees) || 0, 180)));
    if (angle <= 1e-10 || !this.controls) return;
    const target = this.controls.target.clone();
    const radius = this.activeCamera.position.clone().sub(target);
    if (radius.lengthSq() <= 1e-12) return;
    const up = this.activeCamera.up.clone().normalize();
    const viewDirection = target.clone().sub(this.activeCamera.position).normalize();
    const screenRight = new THREE.Vector3().crossVectors(viewDirection, up).normalize();
    let axis;
    let signedAngle;
    if (direction === "right") [axis, signedAngle] = [up, angle];
    else if (direction === "left") [axis, signedAngle] = [up, -angle];
    else if (direction === "up") [axis, signedAngle] = [screenRight, -angle];
    else if (direction === "down") [axis, signedAngle] = [screenRight, angle];
    else if (direction === "roll-left") [axis, signedAngle] = [viewDirection, -angle];
    else if (direction === "roll-right") [axis, signedAngle] = [viewDirection, angle];
    else return;
    const rotation = new THREE.Quaternion().setFromAxisAngle(axis, signedAngle);
    this.activeCamera.up.applyQuaternion(rotation).normalize();
    if (!String(direction).startsWith("roll-")) {
      radius.applyQuaternion(rotation);
      this.activeCamera.position.copy(target).add(radius);
    }
    this.activeCamera.lookAt(target);
    this.controls.update();
    this.render();
  }

  viewAlongCurrentOperation() {
    this.viewAlongCartesianDirection(this.operationViewDirection, this.operationFocusPoint);
  }

  viewAlongCartesianDirection(directionValues, focusValues = null) {
    if (!Array.isArray(directionValues) || !this.controls) return;
    const direction = new THREE.Vector3(...directionValues);
    if (direction.lengthSq() <= 1e-12) return;
    direction.normalize();
    const focus = Array.isArray(focusValues)
      ? new THREE.Vector3(...focusValues)
      : this.controls.target.clone();
    const currentDistance = this.activeCamera.position.distanceTo(this.controls.target);
    const distance = Math.max(currentDistance, this.sceneSpan() * 1.2, 1);
    const referenceUp = Math.abs(direction.z) < 0.92
      ? new THREE.Vector3(0, 0, 1)
      : new THREE.Vector3(0, 1, 0);
    const screenRight = new THREE.Vector3().crossVectors(direction, referenceUp).normalize();
    const up = new THREE.Vector3().crossVectors(screenRight, direction).normalize();
    this.controls.target.copy(focus);
    this.activeCamera.position.copy(focus).addScaledVector(direction, distance);
    this.activeCamera.up.copy(up);
    this.activeCamera.lookAt(focus);
    this.controls.update();
    this.render();
  }

  setCameraCenter(centerValues) {
    if (!Array.isArray(centerValues) || !this.controls) return;
    const center = new THREE.Vector3(...centerValues);
    const offset = center.clone().sub(this.controls.target);
    this.controls.target.copy(center);
    this.activeCamera.position.add(offset);
    this.activeCamera.lookAt(center);
    this.controls.update();
    this.render();
  }

  resetCamera() {
    const saved = this.defaultCameraState;
    if (!saved || !this.controls) return;
    this.activeCamera.position.copy(saved.position);
    this.activeCamera.up.copy(saved.up);
    this.controls.target.copy(saved.target);
    this.activeCamera.lookAt(saved.target);
    this.activeCamera.updateProjectionMatrix();
    this.controls.update();
    this.render();
  }

  setBackgroundMode(mode) {
    const resolved = mode === "light" ? "light" : "dark";
    if (resolved === this.backgroundMode) return;
    this.backgroundMode = resolved;
    this.scene.background.set(resolved === "light" ? 0xf4f6f3 : 0x0b0f14);
    this.render();
  }

  setLegendItems(atoms, styles) {
    const styleByAtom = new Map(styles.map(style => [Number(style.index), style]));
    const items = new Map();
    for (const atom of atoms) {
      if (items.has(atom.element)) continue;
      items.set(atom.element, styleByAtom.get(Number(atom.source_atom ?? atom.index))?.color || "#9aa5b1");
    }
    this.legendItems = [...items.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    this.renderLegend();
  }

  setLegendVisible(visible) {
    if (!this.legend) return;
    this.legend.hidden = !visible;
    this.renderLegend();
  }

  renderLegend() {
    if (!this.legend) return;
    this.legend.replaceChildren();
    for (const [element, color] of this.legendItems) {
      const item = document.createElement("div");
      item.className = "atom-legend-item";
      const swatch = document.createElement("span");
      swatch.className = "atom-legend-swatch";
      swatch.style.backgroundColor = color;
      const label = document.createElement("span");
      label.textContent = element;
      item.append(swatch, label);
      this.legend.appendChild(item);
    }
  }

  addAxis(axis, span) {
    const point = new THREE.Vector3(...axis.point_cart);
    const direction = new THREE.Vector3(...axis.direction_cart).normalize();
    // Molecules are small and their atoms sit on/near the axis, so extend the
    // axis well past the atom cloud to keep it visible. Keep the original
    // slender radius for both molecules and crystals.
    const radius = span * 0.009;
    const length = this.isMolecule ? Math.max(span * 1.8, span + 4 * this.maxAtomRadius) : span * 1.45;
    const geometry = new THREE.CylinderGeometry(radius, radius, length, 16);
    const material = new THREE.MeshBasicMaterial({color: SYMMETRY_ELEMENT_COLORS.axis, transparent: true, opacity: 0.92});
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(point);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    mesh.renderOrder = 3;
    this.addSymmetryObject(mesh);
  }

  // A static curved arrow around an order>2 rotation axis, showing which way it
  // turns. A bare axis line has no handedness, so a Cn (n>2) or Sn operation's
  // two directions look identical without this — e.g. NH3's C3 and C3² both
  // send one H atom to *an*other H, and only the rotation sense tells you which.
  // `axis.rotation_sign` (set server-side, see animation_api.axis_rotation_sign)
  // is null for order<=2 rotations and mirrors/inversions, where there is only
  // one direction to begin with. Swept with THREE's own axis-angle convention,
  // which is the same right-hand rule the server used to compute the sign, so
  // no extra correction for camera or viewing side is needed.
  //
  // Kept deliberately small (a short, thin hint rather than a bold one) — an
  // early version used a wide sweep and a solid 3D cone arrowhead and read as
  // far too prominent next to the axis itself (2026-09-09 follow-up). The
  // arrowhead here is a plain two-line "V", same thin material as the arc,
  // instead of a shaded mesh. Opaque with depthTest disabled (same as
  // addCenter()'s inversion-centre marker below) rather than translucent with
  // normal depth testing: a follow-up report said the translucent version was
  // *harder* to read, and a large structure's atoms can otherwise occlude the
  // indicator entirely since it is drawn at the atoms' own depth.
  addRotationArc(axis, span) {
    if (!axis.rotation_sign) return;
    const point = new THREE.Vector3(...axis.point_cart);
    const direction = new THREE.Vector3(...axis.direction_cart).normalize();
    const helper = Math.abs(direction.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    const start = new THREE.Vector3().crossVectors(direction, helper).normalize();
    const radius = this.isMolecule
      ? Math.max(this.maxAtomRadius * 1.3, span * 0.1)
      : span * 0.14;
    const sweep = THREE.MathUtils.degToRad(55);
    const segments = 14;
    const points = [];
    for (let i = 0; i <= segments; i++) {
      const angle = axis.rotation_sign * sweep * (i / segments);
      points.push(start.clone().applyAxisAngle(direction, angle).multiplyScalar(radius).add(point));
    }
    const material = new THREE.LineBasicMaterial({
      color: SYMMETRY_ELEMENT_COLORS.axis, depthTest: false,
    });
    const arc = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material);
    arc.renderOrder = 4;
    this.addSymmetryObject(arc);

    const tip = points[points.length - 1];
    const back = new THREE.Vector3().subVectors(points[points.length - 2], tip).normalize();
    const headLength = Math.max(radius * 0.3, span * 0.016);
    const headAngle = THREE.MathUtils.degToRad(25);
    const legs = [1, -1].flatMap((sign) => [
      tip,
      back.clone().applyAxisAngle(direction, sign * headAngle).multiplyScalar(headLength).add(tip),
    ]);
    const head = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(legs), material);
    head.renderOrder = 4;
    this.addSymmetryObject(head);
  }

  addPlane(plane, span) {
    const point = new THREE.Vector3(...plane.point_cart);
    const halfWidth = this.isMolecule ? span * 0.65 : span * 0.52;
    const basis1 = new THREE.Vector3(...plane.basis1_cart).normalize().multiplyScalar(halfWidth);
    const basis2 = new THREE.Vector3(...plane.basis2_cart).normalize().multiplyScalar(halfWidth);
    const corners = [
      point.clone().sub(basis1).sub(basis2),
      point.clone().add(basis1).sub(basis2),
      point.clone().add(basis1).add(basis2),
      point.clone().sub(basis1).add(basis2),
    ];
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute([
      ...corners[0].toArray(), ...corners[1].toArray(), ...corners[2].toArray(),
      ...corners[0].toArray(), ...corners[2].toArray(), ...corners[3].toArray(),
    ], 3));
    const material = new THREE.MeshBasicMaterial({
      color: SYMMETRY_ELEMENT_COLORS.planeFill,
      transparent: true,
      opacity: 0.24,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = 1;
    this.addSymmetryObject(mesh);

    const outlineGeometry = new THREE.BufferGeometry().setFromPoints([...corners, corners[0]]);
    const outline = new THREE.Line(
      outlineGeometry,
      new THREE.LineBasicMaterial({color: SYMMETRY_ELEMENT_COLORS.planeOutline, transparent: true, opacity: 0.9}),
    );
    outline.renderOrder = 2;
    this.addSymmetryObject(outline);
  }

  addCenter(center, span) {
    const size = this.isMolecule
      ? Math.max(this.maxAtomRadius * 1.1, span * 0.09)
      : Math.max(span * 0.09, 0.18);
    const boxGeometry = new THREE.BoxGeometry(size, size, size);
    const geometry = new THREE.EdgesGeometry(boxGeometry);
    boxGeometry.dispose();
    const material = new THREE.LineBasicMaterial({
      color: SYMMETRY_ELEMENT_COLORS.center,
      transparent: true,
      opacity: 0.95,
      depthTest: false,
    });
    const cube = new THREE.LineSegments(geometry, material);
    cube.position.set(...center.point_cart);
    cube.renderOrder = 4;
    this.addSymmetryObject(cube);
  }

  addGlideArrow(plane, translation, span) {
    const vector = new THREE.Vector3(...translation);
    if (vector.lengthSq() < 1e-16) return;
    const direction = vector.normalize();
    const length = span * 0.42;
    const origin = new THREE.Vector3(...plane.point_cart).addScaledVector(direction, -length * 0.5);
    const arrow = new THREE.ArrowHelper(
      direction,
      origin,
      length,
      SYMMETRY_ELEMENT_COLORS.glideArrow,
      Math.max(span * 0.07, 0.12),
      Math.max(span * 0.028, 0.055),
    );
    arrow.line.material.transparent = true;
    arrow.line.material.opacity = 0.58;
    arrow.cone.material.transparent = true;
    arrow.cone.material.opacity = 0.62;
    arrow.renderOrder = 4;
    this.addSymmetryObject(arrow);
  }

  addSymmetryObject(object) {
    this.symmetryObjects.push(object);
    this.content.add(object);
  }

  clearSymmetryElements() {
    for (const object of this.symmetryObjects) {
      this.content.remove(object);
      object.traverse(child => {
        child.geometry?.dispose();
        if (Array.isArray(child.material)) child.material.forEach(material => material.dispose());
        else child.material?.dispose();
      });
    }
    this.symmetryObjects = [];
    this.render();
  }

  clearTargetMarkers() {
    for (const object of this.targetMarkerObjects) {
      this.content.remove(object);
      object.geometry?.dispose();
      object.material?.dispose();
    }
    this.targetMarkerObjects = [];
  }

  updateAnimationTargetMarkers() {
    this.clearTargetMarkers();
    if (!this.showAnimationTargets || !this.showAnimationTargetCopies) {
      this.render();
      return;
    }
    const lattice = this.renderData?.unit_cell?.lattice;
    const shiftIndices = [
      [-1, 0, 0],
      [1, 0, 0],
      [0, -1, 0],
      [0, 1, 0],
      [0, 0, -1],
      [0, 0, 1],
    ];
    const targetShifts = Array.isArray(lattice)
      ? shiftIndices.map(shift => new THREE.Vector3(...[
          shift[0] * lattice[0][0] + shift[1] * lattice[1][0] + shift[2] * lattice[2][0],
          shift[0] * lattice[0][1] + shift[1] * lattice[1][1] + shift[2] * lattice[2][1],
          shift[0] * lattice[0][2] + shift[1] * lattice[1][2] + shift[2] * lattice[2][2],
        ]))
      : [new THREE.Vector3(0, 0, 0)];
    const geometry = new THREE.SphereGeometry(1, 20, 14);
    for (const instance of this.atomInstances.values()) {
      if (!this.isAtomVisible(instance)) continue;
      const materialOptions = {
        color: instance.baseColor,
        transparent: true,
        depthWrite: false,
      };
      const startVector = new THREE.Vector3(...instance.start);
      for (const shift of targetShifts) {
        const marker = new THREE.Mesh(geometry.clone(), new THREE.MeshBasicMaterial({
          ...materialOptions,
          opacity: 0.14,
        }));
        marker.position.copy(startVector).add(shift);
        marker.scale.setScalar(instance.radius * 0.92);
        marker.renderOrder = 1;
        this.targetMarkerObjects.push(marker);
        this.content.add(marker);
      }
    }
    geometry.dispose();
    this.render();
  }

  sceneSpan() {
    const minimum = this.renderData?.bounds_min || [-1, -1, -1];
    const maximum = this.renderData?.bounds_max || [1, 1, 1];
    const size = new THREE.Vector3(...maximum).sub(new THREE.Vector3(...minimum));
    return Math.max(size.x, size.y, size.z, 1);
  }

  resetAnimation() {
    this.animationProgress = 0;
    this.animationStartedAt = null;
    this.lastProgressBucket = -1;
    this.lastPublishedProgressBucket = -1;
    this.clearStartMarkers();
    for (const [instanceId, instance] of this.atomInstances) {
      this.setAtomPosition(instanceId, instance.start);
    }
    this.markInstanceMatricesDirty();
    if (this.baseStatus) this.setStatus(`${this.baseStatus}, animation ready`);
    this.render();
    this.publishAnimationProgress(true);
  }

  setAnimationProgress(progress) {
    this.playing = false;
    this.serverPlaying = false;
    this.animationStartedAt = null;
    this.animationProgress = Math.max(0, Math.min(Number(progress) || 0, 1));
    if (this.animationProgress > 0) {
      if (!this.startMarkerObjects.length) this.showStartMarkers();
    } else {
      this.clearStartMarkers();
    }
    this.applyAnimationProgress();
    this.setStatus(`${this.baseStatus}, movement ${Math.round(this.animationProgress * 100)}%`);
    this.publishAnimationProgress(true);
    this.render();
  }

  applyAnimationProgress() {
    for (const [instanceId, instance] of this.atomInstances) {
      const path = this.animationPathForInstance(instance);
      const applies = pathAppliesToDisplayInstance(path, instance);
      const evaluated = applies
        ? evaluatePath(path, this.animationProgress, instance.start)
        : instance.start;
      const position = applies
        ? applyBoundaryContext(evaluated, this.boundaryContext)
        : evaluated;
      this.setAtomPosition(instanceId, position);
    }
    this.markInstanceMatricesDirty();
    this.updateCustomSequenceElements();
  }

  clearTrajectoryLines() {
    for (const object of this.trajectoryObjects) {
      this.content.remove(object);
      object.geometry?.dispose();
      object.material?.dispose();
    }
    this.trajectoryObjects = [];
  }

  updateTrajectoryLines() {
    this.clearTrajectoryLines();
    if (!(this.state.show_trajectories || this.showTrajectories) || !this.animationPaths.size) {
      this.render();
      return;
    }
    const positions = [];
    for (const instance of this.atomInstances.values()) {
      if (!this.isAtomVisible(instance)) continue;
      const path = this.animationPathForInstance(instance);
      if (!pathAppliesToDisplayInstance(path, instance)) continue;
      const samples = new Set([0, 1, ...pathBreakpoints(path, instance.start)]);
      // Wrapped trajectories need dense sampling so each straight segment folds
      // at the cell face instead of jumping across it.
      if (pathHasCurvedMotion(path) || this.boundaryContext?.mode === "wrap") {
        for (let index = 1; index < 49; index += 1) samples.add(index / 48);
      }
      const progressValues = [...samples].sort((a, b) => a - b);
      let previousRaw = evaluatePath(path, progressValues[0], instance.start);
      let previousWrapped = applyBoundaryContext(previousRaw, this.boundaryContext);
      for (const progress of progressValues.slice(1)) {
        const currentRaw = evaluatePath(path, progress, instance.start);
        const currentWrapped = applyBoundaryContext(currentRaw, this.boundaryContext);
        const rawDistanceSquared = currentRaw.reduce((sum, value, axis) => (
          sum + (value - previousRaw[axis]) ** 2
        ), 0);
        const wrappedDistanceSquared = currentWrapped.reduce((sum, value, axis) => (
          sum + (value - previousWrapped[axis]) ** 2
        ), 0);
        // A large wrapped step over a tiny raw step means the segment crossed a
        // cell face; skip that connector so the line folds instead of spanning.
        if (rawDistanceSquared > 1e-16 && wrappedDistanceSquared <= rawDistanceSquared + 1e-6) {
          positions.push(...previousWrapped, ...currentWrapped);
        }
        previousRaw = currentRaw;
        previousWrapped = currentWrapped;
      }
    }
    if (!positions.length) {
      this.render();
      return;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({
      color: 0xf6c85f,
      transparent: true,
      opacity: 0.62,
      depthWrite: false,
    });
    const lines = new THREE.LineSegments(geometry, material);
    lines.renderOrder = 4;
    this.trajectoryObjects.push(lines);
    this.content.add(lines);
    this.render();
  }

  publishAnimationProgress(force = false) {
    const bucket = Math.round(this.animationProgress * 100);
    if (!force && bucket === this.lastPublishedProgressBucket) return;
    this.lastPublishedProgressBucket = bucket;
    window.dispatchEvent(new CustomEvent("symmetry-animation-progress-update", {
      detail: {progress: this.animationProgress},
    }));
  }

  showStartMarkers() {
    this.clearStartMarkers();
    for (const instance of this.atomInstances.values()) {
      if (!this.isAtomVisible(instance)) continue;
      const path = this.animationPathForInstance(instance);
      if (!pathAppliesToDisplayInstance(path, instance)) continue;
      const geometry = new THREE.SphereGeometry(instance.radius * 0.98, 20, 12);
      const material = new THREE.MeshBasicMaterial({
        color: 0xf7dc6f,
        transparent: true,
        opacity: 0.3,
        depthWrite: false,
      });
      const marker = new THREE.Mesh(geometry, material);
      marker.position.fromArray(instance.start);
      marker.renderOrder = 2;
      this.startMarkerObjects.push(marker);
      this.content.add(marker);
    }
    this.container.dataset.startMarkerCount = String(this.startMarkerObjects.length);
    this.render();
  }

  clearStartMarkers() {
    for (const marker of this.startMarkerObjects) {
      this.content.remove(marker);
      marker.geometry.dispose();
      marker.material.dispose();
    }
    this.startMarkerObjects = [];
    this.container.dataset.startMarkerCount = "0";
  }

  updateAnimation(timestamp) {
    if (!this.playing || !this.animationPaths.size) return false;
    if (this.animationStartedAt === null) {
      this.animationStartedAt = timestamp - this.animationProgress * this.animationDurationMs();
    }
    const previousProgress = this.animationProgress;
    this.animationProgress = Math.min(
      (timestamp - this.animationStartedAt) / this.animationDurationMs(),
      1,
    );
    // Recording must run through every boundary without waiting for UI input.
    if (this.state.pause_at_breakpoints && !this.recording) {
      const boundary = this.animationBreakpoints.find(value => (
        value > previousProgress + 1e-9 && value < 1 - 1e-9 && value <= this.animationProgress + 1e-9
      ));
      if (boundary !== undefined) {
        this.animationProgress = boundary;
        this.playing = false;
        this.animationStartedAt = null;
        window.dispatchEvent(new CustomEvent("symmetry-animation-breakpoint-pause", {
          detail: {progress: boundary},
        }));
      }
    }
    this.applyAnimationProgress();
    this.publishAnimationProgress();
    const progressBucket = Math.floor(this.animationProgress * 20);
    if (progressBucket !== this.lastProgressBucket) {
      this.lastProgressBucket = progressBucket;
      this.setStatus(`${this.baseStatus}, animation ${Math.round(this.animationProgress * 100)}%`);
    }
    if (this.animationProgress >= 1) {
      this.playing = false;
      this.animationStartedAt = null;
      this.setStatus(`${this.baseStatus}, animation complete`);
      window.dispatchEvent(new CustomEvent("symmetry-animation-complete"));
      // Without the legacy PyVista window there is nothing server-side to
      // clear the playing flag on completion, so the Start/Stop button would
      // stay on "Stop". Tell the server ourselves (PyVista handles its own).
      if (this.serverPlaying && !this.state.pyvista_enabled) {
        this.serverPlaying = false;
        if (typeof window.symmetryPostState === "function") {
          window.symmetryPostState({playing: false}).catch(() => {});
        }
      }
    }
    return true;
  }

  animationDurationMs() {
    const speed = Math.max(Number(this.state.speed) || 1, 0.1);
    return this.baseAnimationDurationSeconds * 1000 / speed;
  }

  downloadBlob(blob, extension, filename = null) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename ? `${filename}.${extension}` : `${this.defaultDownloadName()}.${extension}`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  defaultDownloadName() {
    // animationOperationIndex is null unless an animation path is loaded, so
    // using it alone saved every recording as "symmetry-op0". The selected
    // operation is what the file actually shows; name it, and the structure
    // too, so exports from a session do not overwrite each other.
    const index = this.state.operation_index ?? this.animationOperationIndex;
    const source = String(this.state.json_path || "").split(/[\\/]/).pop() || "";
    const stem = source.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_-]+/g, "-") || "symmetry";
    return index === null || index === undefined ? stem : `${stem}-op${index}`;
  }

  savePng() {
    this.render();
    this.canvas.toBlob(blob => {
      if (blob) this.downloadBlob(blob, "png");
    }, "image/png");
  }

  async recordGif(filename = null) {
    if (this.recording || !this.animationPaths.size) return;
    this.recording = true;
    const previousProgress = this.animationProgress;
    const durationMs = this.animationDurationMs();
    const frameCount = Math.max(2, Math.min(90, Math.ceil(durationMs / 100) + 1));
    const frameDurationMs = Math.max(20, Math.round(durationMs / (frameCount - 1)));
    const frames = [];
    try {
      this.playing = false;
      this.serverPlaying = false;
      this.animationStartedAt = null;
      this.animationProgress = 0;
      this.showStartMarkers();
      for (let index = 0; index < frameCount; index += 1) {
        this.animationProgress = index / (frameCount - 1);
        this.applyAnimationProgress();
        this.render();
        await new Promise(resolve => requestAnimationFrame(resolve));
        frames.push(this.canvas.toDataURL("image/png"));
      }
      // Keep the completed structure visible for one second before the GIF loops.
      // The final animation frame already occupies one frame duration.
      const extraEndFrames = Math.min(
        180 - frameCount,
        Math.max(0, Math.ceil(GIF_END_HOLD_MS / frameDurationMs) - 1),
      );
      const finalFrame = frames.at(-1);
      for (let index = 0; index < extraEndFrames; index += 1) frames.push(finalFrame);
      const response = await fetch("/api/export_gif", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          frames,
          frame_duration_ms: frameDurationMs,
        }),
      });
      if (!response.ok) throw new Error(await response.text() || `${response.status} ${response.statusText}`);
      this.downloadBlob(await response.blob(), "gif", filename);
    } catch (error) {
      this.setStatus(`GIF export error: ${error.message}`);
    } finally {
      this.recording = false;
      this.playing = false;
      this.animationProgress = previousProgress;
      this.applyAnimationProgress();
      this.render();
    }
  }

  setAtomPosition(instanceId, position) {
    const instance = this.atomInstances.get(Number(instanceId));
    if (!instance) return;
    this.tempPosition.fromArray(position);
    if (this.isAtomVisible(instance)) this.tempScale.setScalar(instance.radius);
    else this.tempScale.copy(ZERO_SCALE);
    this.tempMatrix.compose(this.tempPosition, this.tempQuaternion, this.tempScale);
    instance.mesh.setMatrixAt(instance.instanceIndex, this.tempMatrix);
    instance.current = [...position];
    const marker = this.selectionMarkers.get(Number(instanceId));
    if (marker) {
      marker.visible = this.isAtomVisible(instance);
      marker.position.fromArray(position);
    }
    const pickMarker = this.pickMarkers.get(Number(instanceId));
    if (pickMarker) {
      pickMarker.visible = this.isAtomVisible(instance);
      pickMarker.position.fromArray(position);
    }
  }

  isAtomVisible(instance) {
    return !this.hiddenElements.has(instance.element);
  }

  setElementVisibility(element, visible) {
    if (!element) return;
    if (visible) this.hiddenElements.delete(element);
    else this.hiddenElements.add(element);
    for (const [instanceId, instance] of this.atomInstances) {
      if (instance.element === element) this.setAtomPosition(instanceId, instance.current);
    }
    this.markInstanceMatricesDirty();
    this.updateAtomSelectionHighlight();
    this.clearStartMarkers();
    if (this.animationProgress > 0) this.showStartMarkers();
    this.updateAnimationTargetMarkers();
    this.updateTrajectoryLines();
    this.render();
  }

  markInstanceMatricesDirty() {
    for (const mesh of this.instanceMeshes) {
      mesh.instanceMatrix.needsUpdate = true;
    }
  }

  resize() {
    const width = Math.max(this.container.clientWidth, 1);
    const height = Math.max(this.canvas.clientHeight, 1);
    this.renderer.setSize(width, height, false);
    this.perspectiveCamera.aspect = width / height;
    this.perspectiveCamera.updateProjectionMatrix();
    const span = this.orthographicSpan || ORTHOGRAPHIC_HEIGHT;
    this.orthographicCamera.left = -span * width / height / 2;
    this.orthographicCamera.right = span * width / height / 2;
    this.orthographicCamera.top = span / 2;
    this.orthographicCamera.bottom = -span / 2;
    this.orthographicCamera.updateProjectionMatrix();
    this.controls?.handleResize();
    this.render();
  }

  render() {
    for (const marker of this.selectionMarkers.values()) {
      marker.quaternion.copy(this.activeCamera.quaternion);
    }
    for (const marker of this.pickMarkers.values()) {
      marker.quaternion.copy(this.activeCamera.quaternion);
    }
    this.renderer.render(this.scene, this.activeCamera);
  }

  animate(timestamp) {
    this.frameHandle = null;
    // A hidden view has nothing to show, and both views exist for the whole
    // session: without this the covered one keeps calling controls.update()
    // sixty times a second behind the overlay.
    if (!this.active) return;
    const animationChanged = this.updateAnimation(timestamp);
    this.controls?.update();
    if (animationChanged) this.render();
    this.requestFrame();
  }

  requestFrame() {
    // Guard the handle: setActive(true) twice, or a resume racing the loop,
    // would otherwise leave two chains running and double the frame rate.
    if (this.frameHandle !== null || !this.active) return;
    this.frameHandle = requestAnimationFrame(this.animate);
  }

  setActive(active) {
    const next = Boolean(active);
    if (this.active === next) return;
    this.active = next;
    if (!next) return;
    // The container was display:none while hidden, so the canvas has to be
    // measured again before the first frame back.
    this.resize();
    this.requestFrame();
  }

  setStatus(message) {
    if (!this.status) return;
    const visible = /(error|failed|unavailable|unsupported)/i.test(String(message));
    this.status.textContent = visible ? message : "";
    this.status.hidden = !visible;
  }
}


async function initialize() {
  const container = document.getElementById("three-view");
  if (!container) return;
  const view = new StaticStructureView(container);
  window.symmetryThreeView = view;
  // With --mode puzzle the mode is applied before this module finishes loading,
  // so the entry event has already been and gone; read the mode off the body
  // instead of waiting for one.
  view.setActive(!document.body.classList.contains("in-puzzle"));
  try {
    await view.refresh();
    const stateResponse = await fetch("/api/state");
    if (stateResponse.ok) {
      const initialState = await stateResponse.json();
      await view.syncState(initialState);
      view.lastStateSignature = `${initialState.json_path || ""}|${initialState.reload_request_id || 0}`;
    }
  } catch (error) {
    view.setStatus(`3D view error: ${error.message}`);
  }

  let pollInProgress = false;
  setInterval(async () => {
    if (pollInProgress) return;
    // Puzzle mode drives its own view; pause the analysis poll while it is open
    // (the next tick after returning re-syncs via the signature check). One
    // flag now gates both this poll and the render loop.
    if (!view.active) return;
    pollInProgress = true;
    try {
      const response = await fetch("/api/state");
      if (!response.ok) return;
      const state = await response.json();
      const signature = `${state.json_path || ""}|${state.reload_request_id || 0}`;
      if (signature !== view.lastStateSignature) {
        await view.refresh();
        await view.syncState(state, {structure_reload: true});
      } else {
        await view.syncState(state);
      }
      view.lastStateSignature = signature;
    } catch (error) {
      view.setStatus(`3D view refresh error: ${error.message}`);
    } finally {
      pollInProgress = false;
    }
  }, 1000);

  window.addEventListener("symmetry-state-update", event => {
    const detail = event.detail || {};
    view.syncState(detail.state || {}, detail.update || {}).catch(error => {
      view.setStatus(`3D animation error: ${error.message}`);
    });
  });
}


initialize().catch(error => {
  const status = document.querySelector("[data-three-status]");
  if (status) {
    status.textContent = `3D view initialization error: ${error.message}`;
    status.hidden = false;
  }
  console.error("Three.js initialization failed", error);
});
