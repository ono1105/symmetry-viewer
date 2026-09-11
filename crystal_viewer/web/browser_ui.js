let operations = [];
let atoms = [];
let atomMotionBySource = new Map();
let lastAtomMotionRequestKey = "";
let pendingAtomMotionRequestKey = "";
let atomMotionRequestGeneration = 0;
let lastAtomRenderSignature = "";
let lastOperationRenderSignature = "";
let lastOperationDetailsSignature = "";
let lastStructureInfoSignature = "";
let state = {};
let directionFilterValue = "";
let operationTypeFilterValue = "";
let atomElementFilterValue = "";
let operationLabelMode = "itc_like";
let customOperationSequence = [];
let customSequenceRevision = 0;
let customComposeRequestGeneration = 0;
let summariesReady = false;
let activeMode = "standard";
let experienceMode = "beginner";
let customUnmappedAtoms = new Set();
let sourceKind = "crystal";
let selectedStructureKind = "crystal";
let importInProgress = false;
let refreshInProgress = false;
let analysisBootComplete = false;
let analysisRefreshInProgress = false;
let importRequestId = 0;
let exampleCatalog = {crystal: [], molecule: []};
let selectedExamplePath = "";
let movementBreakpoints = [0, 1];
let movementProgressValue = 0;
const MOVEMENT_FRAME_STEP = 0.02;
const LOOP_RESTART_DELAY_MS = 1000;
let loopRestartTimer = null;
let loopRestartInProgress = false;
let fullscreenOperationOptionsSignature = "";
let selectedAtomDetailsOpen = true;
let displayRangeUpdatePending = false;
const STRUCTURE_KIND_UI = JSON.parse(document.getElementById("structure-kind-config").textContent);
const threeViewPanel = document.getElementById("three-view-panel");
const viewPanel = document.getElementById("view-panel");
const fullscreenControls = document.getElementById("fullscreen-controls");
const fullscreenControlsContent = document.getElementById("fullscreen-controls-content");
const animationPanel = document.getElementById("animation-panel");
const selectedOperationPanel = document.querySelector(".selected-operation-panel");
const utilityPanel = document.getElementById("utility-panel");
const animationPanelPlaceholder = document.createComment("animation-panel-placeholder");
const cameraPanelPlaceholder = document.createComment("camera-panel-placeholder");
const fullscreenAnimationDetails = document.createElement("details");
fullscreenAnimationDetails.className = "fullscreen-animation-details";
const fullscreenAnimationSummary = document.createElement("summary");
fullscreenAnimationSummary.textContent = "Animation";
fullscreenAnimationDetails.append(fullscreenAnimationSummary);
let cameraPanelOpenBeforeFullscreen = false;
// Keep the explanatory matrix directly below the 3D object it describes.
// Display controls are secondary to playback, so they belong below Animation.
document.querySelector(".viewer-primary").appendChild(selectedOperationPanel);
document.querySelector(".side-stack").appendChild(utilityPanel);
animationPanel.before(animationPanelPlaceholder);
viewPanel.before(cameraPanelPlaceholder);

function syncCameraPanelLayout() {
  // Outside full screen Camera belongs to the lower "display and details"
  // area, not over the structure.  It is moved beside the canvas only while
  // full screen controls are open.
  threeViewPanel.classList.toggle("camera-open", false);
}

viewPanel.addEventListener("toggle", syncCameraPanelLayout);
syncCameraPanelLayout();

function moveAnimationPanelIntoFullscreenControls() {
  if (fullscreenAnimationDetails.parentElement === animationPanel) return;
  fullscreenAnimationDetails.append(...animationPanel.childNodes);
  animationPanel.append(fullscreenAnimationDetails);
  fullscreenAnimationDetails.open = true;
}

function restoreAnimationPanelFromFullscreenControls() {
  if (fullscreenAnimationDetails.parentElement !== animationPanel) return;
  animationPanel.replaceChildren(
    ...[...fullscreenAnimationDetails.childNodes].filter(node => node !== fullscreenAnimationSummary),
  );
}

function syncFullscreenControlsLayout() {
  const fullscreen = document.fullscreenElement === threeViewPanel;
  if (fullscreen) {
    fullscreenControlsContent.append(viewPanel, animationPanel);
    moveAnimationPanelIntoFullscreenControls();
    viewPanel.open = fullscreenControls.open;
  } else {
    cameraPanelPlaceholder.after(viewPanel);
    restoreAnimationPanelFromFullscreenControls();
    animationPanelPlaceholder.after(animationPanel);
    fullscreenControls.open = false;
    viewPanel.open = cameraPanelOpenBeforeFullscreen;
  }
  threeViewPanel.classList.toggle("fullscreen-controls-open", fullscreen && fullscreenControls.open);
  syncCameraPanelLayout();
}

fullscreenControls.addEventListener("toggle", () => {
  if (document.fullscreenElement === threeViewPanel) {
    viewPanel.open = fullscreenControls.open;
    syncFullscreenControlsLayout();
  }
});

const fullscreenButton = document.getElementById("toggle-view-fullscreen");

function syncFullscreenButton() {
  const active = document.fullscreenElement === threeViewPanel;
  fullscreenButton.textContent = active ? "Exit full screen" : "Full screen";
  fullscreenButton.setAttribute("aria-label", active ? "Exit full screen view" : "Enter full screen view");
}

fullscreenButton.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement === threeViewPanel) {
      await document.exitFullscreen();
    } else {
      await threeViewPanel.requestFullscreen();
    }
  } catch (error) {
    document.getElementById("status").textContent = `Full screen error: ${error.message || error}`;
  }
});
document.addEventListener("fullscreenchange", () => {
  const entering = document.fullscreenElement === threeViewPanel;
  if (entering) {
    cameraPanelOpenBeforeFullscreen = viewPanel.open;
    fullscreenControls.open = false;
  }
  syncFullscreenButton();
  syncFullscreenControlsLayout();
});
syncFullscreenButton();

async function api(path, options) {
  const response = await fetch(path, options);
  if (response.status === 204) return null;
  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch (_error) {
      body = null;
    }
  }
  if (!response.ok) {
    const message = body && body.error ? body.error : (text || `${response.status} ${response.statusText}`);
    throw new Error(message);
  }
  return body;
}

function optionText(operation) {
  if (sourceKind === "molecule") {
    const symbol = molecularSchoenfliesSymbol(operation);
    return experienceMode === "beginner" ? symbol : `op ${operation.index}: ${symbol}`;
  }
  if (experienceMode === "beginner") return beginnerOperationText(operation);
  const itc = operation.itc_like_summary || operation.itc_coordinate_summary;
  return `op ${operation.index}: ${formatItcLikeSymbol(itc || displayOperationSymbol(operation))}`;
}

function molecularSchoenfliesSymbol(operation) {
  const symbol = String(operation.symbol || operation.display_symbol || "?");
  const mirror = symbol.match(/^sigma(?:_([vhd]))?$/);
  if (mirror) return mirror[1] ? `σ<sub>${mirror[1]}</sub>` : "σ";
  const match = symbol.match(/^([CS])(\\d+|∞)$/);
  return match ? `${match[1]}<sub>${match[2]}</sub>` : formatSymbol(symbol);
}

function beginnerOperationText(operation) {
  const kind = String(operation.kind || "").toLowerCase();
  const reportedAngle = operation.angle_deg;
  const matrixOrder = Number(operation.order);
  const notationOrder = improperNotationOrder(operation);
  const fallbackOrder = isImproperOperation(operation) ? notationOrder : matrixOrder;
  const angle = isImproperOperation(operation) && Number.isFinite(notationOrder) && notationOrder > 1
    ? 360 / notationOrder
    : reportedAngle !== null && reportedAngle !== undefined && Number.isFinite(Number(reportedAngle))
      ? Math.abs(Number(reportedAngle))
      : (Number.isFinite(fallbackOrder) && fallbackOrder > 1 ? 360 / fallbackOrder : 0);
  const angleText = Number.isFinite(angle) && angle > 1e-6
    ? `${Math.round(angle)}°`
    : "";
  const symbol = beginnerOperationSymbol(operation);
  const reading = beginnerOperationReading(operation);
  const label = `${symbol} (${reading})`;
  if (kind === "identity") return `${label}:恒等操作`;
  if (kind.includes("screw")) return `${label}:${angleText}回転+平行移動`;
  if (kind.includes("glide")) return `${label}:鏡映+平行移動`;
  if (kind.includes("mirror")) return `${label}:鏡映`;
  if (kind.includes("inversion") && !kind.includes("roto")) return `${label}:反転`;
  if (kind.includes("improper") || kind.includes("rotoinversion") || kind.includes("rotoreflection")) {
    return `${label}:${angleText}回転+反転`;
  }
  if (kind.includes("rotation")) return `${label}:${angleText}回転`;
  if (kind.includes("translation")) return `${label}:平行移動`;
  return `${label}:対称操作`;
}

function beginnerOperationSymbol(operation) {
  const kind = String(operation.kind || "").toLowerCase();
  const order = improperNotationOrder(operation);
  if (kind.includes("glide")) return "g";
  if (kind.includes("improper") || kind.includes("rotoinversion") || kind.includes("rotoreflection")) {
    return `<span class="overline">${order}</span>`;
  }
  if (kind.includes("inversion")) return '<span class="overline">1</span>';
  return formatSymbol(operation.display_symbol || operation.symbol || "?");
}

function beginnerOperationReading(operation) {
  const kind = String(operation.kind || "").toLowerCase();
  const order = isImproperOperation(operation) ? improperNotationOrder(operation) : (operation.order || "");
  if (kind === "identity") return "恒等";
  if (kind.includes("screw")) return `${order}回らせん`;
  if (kind.includes("glide")) return "映進";
  if (kind.includes("mirror")) return "鏡映";
  if (kind.includes("inversion") && !kind.includes("roto")) return "反転";
  if (kind.includes("improper") || kind.includes("rotoinversion") || kind.includes("rotoreflection")) {
    return `${order}回回反`;
  }
  if (kind.includes("rotation")) return `${order}回回転`;
  if (kind.includes("translation")) return "並進";
  return "対称操作";
}

function operationSummaryText(operation) {
  if (operationLabelMode === "itc_like") {
    return operation.itc_like_summary || operation.itc_coordinate_summary || "";
  }
  return operation.element_summary || "";
}

function renderHtml(text) {
  const template = document.createElement("template");
  template.innerHTML = text;
  return template.content;
}

function setRichSelectOpen(pickerId, open) {
  const picker = document.getElementById(pickerId);
  const trigger = picker.querySelector(".rich-select-trigger");
  const menu = picker.querySelector(".rich-select-menu");
  menu.hidden = !open;
  trigger.setAttribute("aria-expanded", String(open));
}

function installRichSelectToggle(pickerId) {
  const picker = document.getElementById(pickerId);
  picker.querySelector(".rich-select-trigger").addEventListener("click", event => {
    event.stopPropagation();
    const menu = picker.querySelector(".rich-select-menu");
    for (const other of document.querySelectorAll(".rich-select")) {
      setRichSelectOpen(other.id, other === picker && menu.hidden);
    }
  });
}

installRichSelectToggle("example-picker");
installRichSelectToggle("cop-operation-picker");
document.addEventListener("click", () => {
  for (const picker of document.querySelectorAll(".rich-select")) setRichSelectOpen(picker.id, false);
});

function formatSymbol(symbol) {
  const subscripts = "₀₁₂₃₄₅₆";
  return String(symbol)
    .replace(/sigma/g, "σ")
    .replace(/_([0-6])/g, (_, digit) => subscripts[Number(digit)]);
}

function formatItcLikeSymbol(symbol) {
  return formatSymbol(symbol)
    .replace(/^\s*-([0-9]+)/, '<span class="overline">$1</span>')
    .replace(/-((?:\d+)?[xyz])/g, '<span class="overline">$1</span>')
    .replace(/^(\s*\d(?:[₀₁₂₃₄₅₆])?)([+-])/, '$1<sup>$2</sup>');
}

function formatPlainOverbar(symbol) {
  return String(symbol).replace(/-([0-9]+)/g, (_, digits) =>
    [...digits].map(digit => `${digit}\u0305`).join(""));
}

function stripHtmlWithOverbars(value) {
  const withUnicodeOverbars = String(value).replace(
    /<span class="overline">([^<]+)<\/span>/g,
    (_, text) => [...text].map(character => `${character}\u0305`).join(""),
  );
  return stripHtml(withUnicodeOverbars);
}

function formatGroupSymbol(symbol) {
  const text = String(symbol);
  const schoenflies = text.match(/^([CDS])(\d+|∞)([a-z]*)$/);
  if (schoenflies) return `${schoenflies[1]}<sub>${schoenflies[2]}${schoenflies[3]}</sub>`;
  return text
    .replace(/sigma/g, "σ")
    .replace(/_([A-Za-z0-9∞]+)/g, "<sub>$1</sub>")
    .replace(/-([0-9]+)/g, '<span class="overline">$1</span>');
}

function resolvedImproperMode() {
  const requested = state.improper_mode || "auto";
  if (requested === "rotoreflection" || requested === "rotoinversion") return requested;
  return sourceKind === "molecule" ? "rotoreflection" : "rotoinversion";
}

function isImproperOperation(operation) {
  const kind = String(operation.kind || "");
  return kind.includes("improper") || kind.includes("rotoinversion") || kind.includes("rotoreflection");
}

function improperNotationOrder(operation) {
  const suppliedOrder = Number(operation.notation_order);
  if (Number.isFinite(suppliedOrder) && suppliedOrder > 0) return suppliedOrder;
  const symbol = String(operation.symbol || operation.display_symbol || "");
  const match = symbol.match(/[0-9]+/);
  if (match) return Number(match[0]);
  const matrixOrder = Number(operation.order);
  return Number.isFinite(matrixOrder) ? matrixOrder : 0;
}

function displayOperationSymbol(operation) {
  if (!isImproperOperation(operation)) {
    return operation.display_symbol || operation.symbol || "";
  }
  const notationOrder = improperNotationOrder(operation);
  if (resolvedImproperMode() === "rotoreflection") {
    const reflectionOrder = operation.order || notationOrder;
    return reflectionOrder ? `S${reflectionOrder}` : (operation.display_symbol || operation.symbol || "");
  }
  return notationOrder
    ? `<span class="overline">${notationOrder}</span>`
    : (operation.display_symbol || operation.symbol || "");
}

function atomDisplayLabel(atom) {
  if (sourceKind !== "crystal" || atom.asymmetric_index === null || atom.asymmetric_index === undefined) {
    return atom.element;
  }
  const siteIndices = [...new Set(
    atoms
      .filter(item => item.element === atom.element && item.asymmetric_index !== null && item.asymmetric_index !== undefined)
      .map(item => Number(item.asymmetric_index))
  )].sort((a, b) => a - b);
  const ordinal = siteIndices.indexOf(Number(atom.asymmetric_index));
  if (siteIndices.length <= 1) return atom.element;
  return ordinal >= 0 ? `${atom.element}${ordinal + 1}` : atom.element;
}

function atomMotionParts(atom) {
  const motion = atomMotionBySource.get(atom.index);
  if (!motion) return null;
  const target = motion.target_atom === null || motion.target_atom === undefined ? "?" : motion.target_atom;
  if (sourceKind === "crystal" && Array.isArray(motion.start_frac) && Array.isArray(motion.target_frac)) {
    return {
      path: `${formatVector(motion.start_frac, formatFrac)} → ${formatVector(motion.target_frac, formatFrac)}`,
      target,
    };
  }
  if (Array.isArray(motion.start_cart) && Array.isArray(motion.target_cart)) {
    return {
      path: `${formatVector(motion.start_cart, formatCoord)} → ${formatVector(motion.target_cart, formatCoord)}`,
      target,
    };
  }
  return {path: "", target};
}

function formatVector(values, formatter) {
  return `(${values.map(formatter).join(", ")})`;
}

function formatFrac(value) {
  return fmtFrac(value);
}

function formatCoord(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3).replace(/\\.?0+$/, "") : "?";
}

function normalizeColor(value, fallback = "#9aa5b1") {
  const text = String(value || "").trim();
  if (/^#[0-9a-fA-F]{6}$/.test(text)) return text.toLowerCase();
  return fallback;
}

function atomEffectiveColor(atom) {
  const atomColors = state.atom_colors || {};
  const elementColors = state.element_colors || {};
  return normalizeColor(
    atomColors[String(atom.index)] || elementColors[atom.element] || atom.default_color,
  );
}

function atomVisible(atom) {
  const atomHidden = state.atom_hidden || {};
  const elementHidden = state.element_hidden || {};
  return !elementHidden[atom.element] && !atomHidden[String(atom.index)];
}

function atomVisibilityUpdate(atom, visible) {
  const atomHidden = Object.assign({}, state.atom_hidden || {});
  const elementHidden = Object.assign({}, state.element_hidden || {});
  if (visible) {
    atomHidden[String(atom.index)] = true;
  } else {
    delete atomHidden[String(atom.index)];
    if (elementHidden[atom.element]) {
      delete elementHidden[atom.element];
      for (const peer of atoms) {
        if (peer.element === atom.element && peer.index !== atom.index) {
          atomHidden[String(peer.index)] = true;
        }
      }
    }
  }
  return {atom_hidden: atomHidden, element_hidden: elementHidden, playing: false, reset: true};
}

function resetAtomAppearance() {
  return postState({
    element_colors: {},
    atom_colors: {},
    element_hidden: {},
    atom_hidden: {},
    playing: false,
    reset: true,
  });
}

function visibilityButton(visible, title, onToggle) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `visibility-toggle${visible ? "" : " hidden"}`;
  button.textContent = visible ? "Visible" : "Hidden";
  button.title = title;
  button.setAttribute("aria-label", title);
  button.addEventListener("click", onToggle);
  return button;
}

function buildOperationRow(operation) {
  const text = optionText(operation);
  const row = document.createElement("button");
  row.type = "button";
  row.className = "operation-row";
  row.dataset.index = operation.index;
  row.setAttribute("role", "option");
  row.setAttribute("aria-selected", operation.index === state.operation_index ? "true" : "false");
  if (operation.index === state.operation_index) row.classList.add("selected");
  row.appendChild(renderHtml(text));
  row.addEventListener("click", () => {
    activeMode = "standard";
    customUnmappedAtoms = new Set();
    invalidateCustomSequenceWork();
    syncActiveModeControls();
    postState({
      active_mode: "standard",
      operation_index: operation.index,
      playing: false,
      reset: true,
      clear_custom_check: true,
    });
  });
  return row;
}

function renderOperations() {
  renderOperationTypeFilter();
  const root = document.getElementById("operations");
  const sorted = sortedOperations();
  renderFullscreenOperationSelector(sorted);
  const filtered = sorted.filter(operation =>
    !(operationTypeFilterValue && operationSymbolFilterKey(operation) !== operationTypeFilterValue)
    && !(directionFilterValue && operationFilterKey(operation) !== directionFilterValue)
  );
  renderOperationCount(filtered.length, sorted.length);
  // Driven by the 1 s poll and by every postState, so most calls have nothing
  // to change. Rebuilding the list anyway discards scroll position and hover.
  const signature = operationListRenderSignature(sorted);
  if (signature === lastOperationRenderSignature && root.childElementCount) return;
  lastOperationRenderSignature = signature;
  root.innerHTML = "";
  for (const operation of filtered) {
    root.appendChild(buildOperationRow(operation));
  }
  if (activeMode === "custom" && experienceMode === "advanced" && sourceKind === "crystal") {
    renderCustomSequenceControls(sorted);
  }
  renderSelectedOperationOverlay();
}

function renderSelectedOperationOverlay() {
  const overlay = document.getElementById("selected-operation-overlay");
  if (!overlay) return;
  if (!structureLoadedForSelectedKind()) {
    overlay.hidden = true;
    overlay.replaceChildren();
    return;
  }
  const operation = operations.find(item => Number(item.index) === Number(state.operation_index));
  const label = activeMode === "custom"
    ? "Custom operation"
    : (operation ? stripHtmlWithOverbars(optionText(operation)) : "Operation");
  overlay.replaceChildren();
  const caption = document.createElement("span");
  caption.className = "selected-operation-overlay-label";
  caption.textContent = "選択中";
  const value = document.createElement("strong");
  value.textContent = label;
  overlay.append(caption, value);
  overlay.hidden = false;
}

function renderFullscreenOperationSelector(sorted) {
  const select = document.getElementById("fullscreen-operation-select");
  if (!select) return;
  const signature = JSON.stringify({
    source: state.json_path || "",
    experienceMode,
    operations: sorted.map(operation => [operation.index, optionText(operation)]),
  });
  if (signature !== fullscreenOperationOptionsSignature) {
    fullscreenOperationOptionsSignature = signature;
    select.replaceChildren();
    for (const operation of sorted) {
      const option = document.createElement("option");
      option.value = String(operation.index);
      option.textContent = stripHtmlWithOverbars(optionText(operation));
      select.appendChild(option);
    }
  }
  select.value = String(state.operation_index);
}

function renderOperationCount(visibleCount, totalCount) {
  const output = document.getElementById("operations-count");
  if (!output) return;
  output.value = `${visibleCount} / ${totalCount}`;
  output.textContent = `${visibleCount} / ${totalCount}`;
}

function operationSymbolFilterKey(operation) {
  if (sourceKind === "molecule") return stripHtml(molecularSchoenfliesSymbol(operation));
  const kind = String(operation.kind || "").toLowerCase();
  if (kind.includes("glide")) return "g";
  if (kind.includes("improper") || kind.includes("rotoinversion") || kind.includes("rotoreflection")) {
    return `bar-${improperNotationOrder(operation)}`;
  }
  if (kind.includes("inversion")) return "bar-1";
  return sortableSymbol(operation);
}

function operationSymbolFilterLabel(operation) {
  if (sourceKind === "molecule") return molecularSchoenfliesSymbol(operation);
  return beginnerOperationSymbol(operation);
}

function operationFilterOrderRank(operation) {
  // Groups operations by symmetry order after grouping by type, so e.g. C2
  // sorts before C3 within rotations, and a finite order sorts before an
  // unresolved/infinite one (C∞, S∞) within the same type.
  if (isImproperOperation(operation)) {
    const order = improperNotationOrder(operation);
    return order > 0 ? order : Infinity;
  }
  const order = Number(operation.order);
  return Number.isFinite(order) && order > 0 ? order : Infinity;
}

function renderOperationTypeFilter() {
  const root = document.getElementById("operation-type-filter");
  if (!root) return;
  const symbols = new Map();
  const ranks = new Map();
  for (const operation of operations) {
    const key = operationSymbolFilterKey(operation);
    if (!symbols.has(key)) {
      symbols.set(key, operationSymbolFilterLabel(operation));
      ranks.set(key, [beginnerOperationTypeRank(operation), operationFilterOrderRank(operation)]);
    }
  }
  const keys = [...symbols.keys()].sort((a, b) =>
    compareNumber(ranks.get(a)[0], ranks.get(b)[0])
    || compareNumber(ranks.get(a)[1], ranks.get(b)[1])
    || compareText(a, b)
  );
  if (operationTypeFilterValue && !keys.includes(operationTypeFilterValue)) operationTypeFilterValue = "";
  root.innerHTML = "";
  const allLabel = "All";
  for (const [key, label] of [["", allLabel], ...keys.map(key => [key, symbols.get(key)])]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `direction-chip${key === operationTypeFilterValue ? " selected" : ""}`;
    button.appendChild(renderHtml(label));
    button.addEventListener("click", () => {
      operationTypeFilterValue = key;
      renderOperations();
    });
    root.appendChild(button);
  }
}

function renderCustomSequenceControls(sorted = sortedOperations()) {
  const select = document.getElementById("cop-operation-select");
  const menu = document.getElementById("cop-operation-menu");
  const list = document.getElementById("custom-sequence-list");
  if (!select || !list) return;
  const selectedValue = select.value;
  select.innerHTML = "";
  menu.innerHTML = "";
  const customSorted = [...sorted].sort((a, b) =>
    compareNumber(beginnerOperationTypeRank(a), beginnerOperationTypeRank(b))
    || compareNumber(Number(a.order) || 0, Number(b.order) || 0)
    || compareNumber(a.index, b.index)
  );
  for (const operation of customSorted) {
    const option = document.createElement("option");
    option.value = String(operation.index);
    option.textContent = stripHtmlWithOverbars(optionText(operation));
    select.appendChild(option);
    const item = document.createElement("button");
    item.type = "button";
    item.className = "rich-select-option";
    item.dataset.value = String(operation.index);
    item.appendChild(renderHtml(optionText(operation)));
    item.addEventListener("click", () => {
      select.value = item.dataset.value;
      select.dispatchEvent(new Event("change"));
      syncCustomOperationPicker();
      setRichSelectOpen("cop-operation-picker", false);
    });
    menu.appendChild(item);
  }
  if (selectedValue && [...select.options].some(option => option.value === selectedValue)) {
    select.value = selectedValue;
  } else if (state.operation_index !== undefined) {
    select.value = String(state.operation_index);
  }
  syncCustomOperationPicker();

  list.innerHTML = "";
  const operationsByIndex = new Map(operations.map(operation => [operation.index, operation]));
  if (customOperationSequence.length) {
    customOperationSequence.forEach((sequenceItem, position) => {
      const operation = sequenceItem.type === "operation" ? operationsByIndex.get(sequenceItem.index) : null;
      const row = document.createElement("div");
      row.className = "sequence-item";
      const label = document.createElement("span");
      label.textContent = `${position + 1}. ${operation ? stripHtmlWithOverbars(optionText(operation)) : customSequenceItemLabel(customOperationSequence[position])}`;
      const moveUp = document.createElement("button");
      moveUp.type = "button";
      moveUp.className = "secondary";
      moveUp.textContent = "↑";
      moveUp.title = "Move earlier";
      moveUp.setAttribute("aria-label", "Move earlier");
      moveUp.disabled = position === 0;
      moveUp.addEventListener("click", () => {
        [customOperationSequence[position - 1], customOperationSequence[position]] = [
          customOperationSequence[position], customOperationSequence[position - 1],
        ];
        invalidateCustomSequenceWork();
        renderCustomSequenceControls();
      });
      const moveDown = document.createElement("button");
      moveDown.type = "button";
      moveDown.className = "secondary";
      moveDown.textContent = "↓";
      moveDown.title = "Move later";
      moveDown.setAttribute("aria-label", "Move later");
      moveDown.disabled = position === customOperationSequence.length - 1;
      moveDown.addEventListener("click", () => {
        [customOperationSequence[position], customOperationSequence[position + 1]] = [
          customOperationSequence[position + 1], customOperationSequence[position],
        ];
        invalidateCustomSequenceWork();
        renderCustomSequenceControls();
      });
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "secondary";
      remove.textContent = "Remove";
      remove.addEventListener("click", () => {
        customOperationSequence.splice(position, 1);
        invalidateCustomSequenceWork();
        renderCustomSequenceControls();
      });
      row.appendChild(label);
      row.appendChild(moveUp);
      row.appendChild(moveDown);
      row.appendChild(remove);
      list.appendChild(row);
    });
  }
}

function syncCustomOperationPicker() {
  const select = document.getElementById("cop-operation-select");
  const operation = operations.find(item => String(item.index) === select.value);
  const trigger = document.getElementById("cop-operation-display");
  trigger.replaceChildren();
  if (operation) trigger.appendChild(renderHtml(optionText(operation)));
  else trigger.textContent = "Select operation";
  for (const item of document.querySelectorAll("#cop-operation-menu .rich-select-option")) {
    item.classList.toggle("selected", item.dataset.value === select.value);
  }
}

function customSequenceItemLabel(item) {
  if (!item) return "unknown";
  if (item.type === "operation") return `op ${item.index}`;
  return item.label || "custom";
}

function hideCustomSequenceResult() {
  // A composition result is valid only for the exact sequence that produced
  // it.  Hiding the message alone left its matrix available to Start after an
  // add/remove/reorder, which replayed the old sequence.
  copMatrix = null;
  const div = document.getElementById("cop-result");
  if (div) div.hidden = true;
}

function invalidateCustomSequenceWork() {
  customSequenceRevision += 1;
  customComposeRequestGeneration += 1;
  hideCustomSequenceResult();
}

function customSequenceContextIsCurrent(revision, structureKey) {
  return revision === customSequenceRevision
    && activeMode === "custom"
    && structureKey === `${state.json_path || ""}|${state.reload_request_id || ""}`;
}

function renderDirectionFilter() {
  const root = document.getElementById("direction-filter");
  const filterType = operationFilterType();
  document.getElementById("operation-filter-label").textContent =
    filterType === "symbol" ? "Symbol" : "Direction";
  const values = [...new Map(
    operations
      .map(operation => [operationFilterKey(operation), operationFilterLabel(operation)])
      .filter(([value]) => value && value !== "none")
  ).entries()].sort((a, b) => compareText(a[1], b[1]));
  root.innerHTML = "";
  const allButton = document.createElement("button");
  allButton.type = "button";
  allButton.className = `direction-chip${directionFilterValue ? "" : " selected"}`;
  allButton.textContent = "All";
  allButton.addEventListener("click", () => {
    directionFilterValue = "";
    renderDirectionFilter();
    renderOperations();
  });
  root.appendChild(allButton);
  for (const [value, label] of values) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `direction-chip${value === directionFilterValue ? " selected" : ""}`;
    button.appendChild(renderHtml(label));
    button.addEventListener("click", () => {
      directionFilterValue = value;
      renderDirectionFilter();
      renderOperations();
    });
    root.appendChild(button);
  }
  if (directionFilterValue && !values.some(([value]) => value === directionFilterValue)) {
    directionFilterValue = "";
    renderDirectionFilter();
    renderOperations();
  }
}

function syncOperationLabelModeControls() {
  const block = document.getElementById("operation-label-mode-block");
  if (block) block.hidden = sourceKind !== "crystal";
  const select = document.getElementById("operation-label-mode");
  if (select) select.value = operationLabelMode;
}

function sortedOperations() {
  if (experienceMode === "beginner") {
    // Simple mode is organized by the familiar operation types.  The internal
    // operation index remains available in Full mode, where it is meaningful.
    return [...operations].sort((a, b) =>
      compareNumber(beginnerOperationTypeRank(a), beginnerOperationTypeRank(b))
      || compareNumber(Number(a.order) || 0, Number(b.order) || 0)
      || compareNumber(Number(a.angle_deg) || 0, Number(b.angle_deg) || 0)
      || compareNumber(a.index, b.index)
    );
  }
  const mode = document.getElementById("operation-sort").value;
  return [...operations].sort((a, b) => {
    if (mode === "symbol") {
      return compareText(sortableSymbol(a), sortableSymbol(b)) || compareNumber(a.index, b.index);
    }
    if (mode === "element") {
      return compareText(a.element_sort_key || "", b.element_sort_key || "") || compareNumber(a.index, b.index);
    }
    if (mode === "direction") {
      return compareText(a.direction_sort_key || "", b.direction_sort_key || "") || compareNumber(a.index, b.index);
    }
    return compareNumber(a.index, b.index);
  });
}

function beginnerOperationTypeRank(operation) {
  const kind = String(operation.kind || "").toLowerCase();
  if (kind === "identity") return 0;
  if (kind.includes("rotation") && !kind.includes("improper")) return 1;
  if (kind.includes("screw")) return 2;
  if (kind.includes("mirror")) return 3;
  if (kind.includes("glide")) return 4;
  if (kind.includes("inversion") && !kind.includes("roto")) return 5;
  if (kind.includes("improper") || kind.includes("rotoinversion") || kind.includes("rotoreflection")) return 6;
  if (kind.includes("translation")) return 7;
  return 8;
}

function sortableSymbol(operation) {
  return stripHtml(`${displayOperationSymbol(operation)}`).replace(/_/g, "");
}

function operationFilterType() {
  if (sourceKind === "molecule") return "symbol";
  return document.getElementById("operation-sort").value === "symbol" ? "symbol" : "direction";
}

function operationFilterKey(operation) {
  if (operationFilterType() === "symbol") return sortableSymbol(operation);
  return operation.direction_sort_key || "";
}

function operationFilterLabel(operation) {
  if (operationFilterType() === "symbol") return formatSymbol(displayOperationSymbol(operation));
  return operation.direction_label || operation.direction_filter_label || operation.direction_sort_key || "";
}

function stripHtml(text) {
  const template = document.createElement("template");
  template.innerHTML = text;
  return template.content.textContent || "";
}

function compareText(a, b) {
  return String(a).localeCompare(String(b), undefined, {numeric: true, sensitivity: "base"});
}

function compareNumber(a, b) {
  return Number(a) - Number(b);
}

function basename(path) {
  const text = String(path || "");
  if (!text) return "-";
  const parts = text.split(/[\\\\/]/);
  return parts[parts.length - 1] || text;
}

function structureDisplayName(path) {
  return basename(path).replace(/\.(?:cif|xyz|json)$/i, "");
}

function formatLength(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(4)} Å` : "-";
}

function formatAngle(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(3)}°` : "-";
}

function structureLoadedForSelectedKind() {
  const loaded = state.structure_loaded !== false && sourceKind !== "empty";
  return loaded && sourceKind === selectedStructureKind;
}

function sourceKindConfig(kind) {
  return STRUCTURE_KIND_UI[kind] || STRUCTURE_KIND_UI.crystal;
}

function setDefaultOperationSortForSourceKind() {
  document.getElementById("operation-sort").value = "index";
  directionFilterValue = "";
}

const CRYSTAL_POINT_GROUP_LABELS = {
  "1": "C_1",
  "-1": "C_i",
  "2": "C_2",
  "m": "C_s",
  "2/m": "C_2h",
  "222": "D_2",
  "mm2": "C_2v",
  "mmm": "D_2h",
  "4": "C_4",
  "-4": "S_4",
  "4/m": "C_4h",
  "422": "D_4",
  "4mm": "C_4v",
  "-42m": "D_2d",
  "4/mmm": "D_4h",
  "3": "C_3",
  "-3": "S_6",
  "32": "D_3",
  "3m": "C_3v",
  "-3m": "D_3d",
  "6": "C_6",
  "-6": "C_3h",
  "6/m": "C_6h",
  "622": "D_6",
  "6mm": "C_6v",
  "-6m2": "D_3h",
  "6/mmm": "D_6h",
  "23": "T",
  "m-3": "T_h",
  "432": "O",
  "-43m": "T_d",
  "m-3m": "O_h",
};

function renderStructureInfo() {
  const panel = document.getElementById("structure-info-panel");
  const root = document.getElementById("structure-info");
  const headerSummary = document.getElementById("header-structure-summary");
  if (!structureLoadedForSelectedKind()) {
    panel.hidden = true;
    root.innerHTML = "";
    headerSummary.hidden = true;
    headerSummary.textContent = "";
    lastStructureInfoSignature = "";
    return;
  }
  const metadata = state.metadata || {};
  const signature = JSON.stringify({
    experienceMode,
    sourceKind,
    selectedStructureKind,
    sourceFile: metadata.source_file || state.json_path,
    formula: metadata.formula,
    symmetryLabel: metadata.symmetry_label,
    pointGroupLabel: metadata.point_group_label,
    operationCount: metadata.operation_count || operations.length,
    latticeParameters: metadata.lattice_parameters,
    displayLatticeParameters: metadata.display_lattice_parameters,
    cellSettingMode: state.cell_setting_mode,
    atoms: atoms.map(atom => [atom.index, atom.element, atom.frac, atom.cart]),
    atomMotion: [...atomMotionBySource.entries()].map(([index, motion]) => [
      index, motion.target_frac, motion.target_cart, motion.stages,
    ]),
  });
  if (signature === lastStructureInfoSignature && !panel.hidden) return;
  const previousDetails = root.querySelector(".atom-position-details");
  const atomPositionsWereOpen = Boolean(previousDetails?.open);
  const atomPositionScrollTop = previousDetails?.querySelector(".atom-position-list")?.scrollTop || 0;
  lastStructureInfoSignature = signature;
  const config = sourceKindConfig(sourceKind);
  const operationCount = metadata.operation_count || operations.length || 0;
  const pointGroup = symmetryLabelWithGenerators(
    metadata.point_group_label || metadata.symmetry_label || "-",
    metadata.point_group_generators,
    {pointGroup: true, groupSymbol: sourceKind !== "crystal"},
  );
  const summaryItems = [
    ["構造", structureDisplayName(metadata.source_file || state.json_path)],
    ["化学式", metadata.formula || "-"],
    ["点群", pointGroup],
    ["操作", operationCount],
  ];
  root.innerHTML = "";
  appendSummaryGrid(root, summaryItems, "primary");
  headerSummary.textContent = `${structureDisplayName(metadata.source_file || state.json_path)} · ${metadata.formula || "-"} · ${stripHtmlWithOverbars(pointGroup)} · ${operationCount}操作`;
  headerSummary.hidden = false;

  if (experienceMode === "advanced") {
    const details = document.createElement("details");
    details.className = "structure-details";
    details.open = false;
    const detailsSummary = document.createElement("summary");
    detailsSummary.textContent = "格子・原子座標";
    details.appendChild(detailsSummary);
    const body = document.createElement("div");
    body.className = "structure-details-body";
    if (sourceKind === "crystal") {
      appendSummaryGrid(body, [
        [config.symmetryLabel, symmetryLabelWithGenerators(metadata.symmetry_label, metadata.space_group_generators, {spaceGroup: true})],
        ["Bravais lattice", bravaisLatticeLabel(metadata.symmetry_label)],
        ["Atoms", metadata.display_atom_count || atoms.length || "-"],
      ], "cell-setting");
      const lattice = metadata.display_lattice_parameters || metadata.lattice_parameters;
      if (lattice) appendLatticeSummary(body, lattice);
    }
    appendAtomPositionSummary(body, atomPositionsWereOpen, atomPositionScrollTop);
    details.appendChild(body);
    root.appendChild(details);
  }
  panel.hidden = false;
}

function appendLatticeSummary(root, lattice) {
  appendSummaryGrid(root, [
    ["a", formatLength(lattice.a)],
    ["b", formatLength(lattice.b)],
    ["c", formatLength(lattice.c)],
    ["alpha", formatAngle(lattice.alpha)],
    ["beta", formatAngle(lattice.beta)],
    ["gamma", formatAngle(lattice.gamma)],
  ], "compact lattice-summary");
}

function crystalSystemFromSymmetryLabel(label) {
  const match = String(label || "").match(/^([0-9]+)/);
  if (!match) return "-";
  const number = Number(match[1]);
  if (number <= 2) return "Triclinic";
  if (number <= 15) return "Monoclinic";
  if (number <= 74) return "Orthorhombic";
  if (number <= 142) return "Tetragonal";
  if (number <= 167) return "Trigonal";
  if (number <= 194) return "Hexagonal";
  if (number <= 230) return "Cubic";
  return "-";
}

function appendAtomPositionSummary(root, open = false, scrollTop = 0) {
  const counts = new Map();
  for (const atom of atoms) counts.set(atom.element, (counts.get(atom.element) || 0) + 1);
  const composition = [...counts.entries()].map(([element, count]) => `${element} × ${count}`).join("、") || "-";
  appendSummaryGrid(root, [["Atoms", composition]], "cell-setting");

  const details = document.createElement("details");
  details.className = "atom-position-details";
  details.open = open;
  const summary = document.createElement("summary");
  summary.textContent = `Atom positions (${atoms.length})`;
  details.appendChild(summary);
  const list = document.createElement("div");
  list.className = "atom-position-list";
  for (const atom of atoms) {
    const isCrystal = sourceKind === "crystal" && Array.isArray(atom.frac);
    const formatter = isCrystal ? formatFrac : formatCoord;
    const start = isCrystal ? atom.frac : (atom.cart || []);
    const motion = atomMotionBySource.get(atom.index);
    const target = isCrystal ? motion?.target_frac : motion?.target_cart;
    const stages = (motion?.stages || [])
      .map(stage => isCrystal ? stage.frac : stage.cart)
      .filter(Array.isArray);
    const coordinates = `(${start.map(formatter).join(", ")})`;
    const stageCoordinates = stages
      .map(stage => ` → (${stage.map(formatter).join(", ")})`)
      .join("");
    const targetCoordinates = Array.isArray(target)
      ? ` → (${target.map(formatter).join(", ")})`
      : "";
    const unit = isCrystal ? (experienceMode === "advanced" ? " fractional" : "") : " Å";
    const row = document.createElement("div");
    row.textContent = `${atom.index + 1}. ${atomDisplayLabel(atom)}: ${coordinates}${stageCoordinates}${targetCoordinates}${unit}`;
    list.appendChild(row);
  }
  details.appendChild(list);
  root.appendChild(details);
  list.scrollTop = scrollTop;
}

function formatCellSettingLabel(value) {
  const mode = String(value || "native").toLowerCase();
  if (mode === "primitive") return "Primitive cell";
  if (mode === "conventional") return "Bravais cell";
  if (mode === "refined") return "Refined";
  if (mode === "hexagonal_conventional") return "Hexagonal conventional";
  return "As loaded";
}

function bravaisLatticeLabel(symmetryLabel) {
  const text = String(symmetryLabel || "").trim();
  const symbol = text.replace(/^\d+\s+/, "");
  const centering = (symbol.match(/^[PABCIFR]/) || ["P"])[0];
  const systemName = crystalSystemFromSymmetryLabel(symmetryLabel);
  const system = systemName.toLowerCase();
  const centeringName = {
    P: "Primitive", A: "Base-centered", B: "Base-centered", C: "Base-centered",
    I: "Body-centered", F: "Face-centered", R: "Rhombohedral",
  }[centering] || centering;
  if (centering === "R") return "Rhombohedral";
  return `${centeringName} ${system}`;
}

function symmetryLabelWithGenerators(label, generators, options = {}) {
  const base = options.spaceGroup
    ? formatSpaceGroupLabel(label)
    : options.pointGroup
      ? formatCrystalPointGroupLabel(label)
      : options.groupSymbol
        ? formatGroupSymbol(label || "-")
        : formatSymbol(label || "-");
  const generatorSet = formatGeneratorSet(generators);
  return generatorSet ? `${base} ${generatorSet}` : base;
}

function formatSpaceGroupLabel(label) {
  const text = String(label || "-");
  const match = text.match(/^([0-9]+)\\s+(.+)$/);
  if (!match) return formatGroupSymbol(text);
  return `No. ${match[1]} ${formatGroupSymbol(match[2])}`;
}

function formatCrystalPointGroupLabel(label) {
  const text = String(label || "-").trim();
  const schoenflies = CRYSTAL_POINT_GROUP_LABELS[text];
  return schoenflies ? `${formatGroupSymbol(schoenflies)} (${formatGroupSymbol(text)})` : formatGroupSymbol(text);
}

function formatGeneratorSet(generators) {
  if (!Array.isArray(generators) || !generators.length) return "";
  const values = generators.filter((value) => value && value !== "identity only");
  if (!values.length) return "";
  return `&lang;${values.map((value) => formatSymbol(value)).join(", ")}&rang;`;
}

function appendSummaryGrid(root, items, extraClass = "") {
  const grid = document.createElement("div");
  grid.className = extraClass ? `summary-grid ${extraClass}` : "summary-grid";
  for (const [label, value] of items) {
    grid.appendChild(summaryItem(label, value));
  }
  root.appendChild(grid);
}

function summaryItem(label, value) {
  const item = document.createElement("div");
  item.className = "summary-item";
  const labelEl = document.createElement("div");
  labelEl.className = "summary-label";
  labelEl.textContent = label;
  const valueEl = document.createElement("div");
  valueEl.className = "summary-value";
  valueEl.appendChild(renderHtml(String(value)));
  item.appendChild(labelEl);
  item.appendChild(valueEl);
  return item;
}

function syncStructureKindButtons() {
  const button = document.getElementById("structure-kind-toggle");
  const moleculeSelected = selectedStructureKind === "molecule";
  button.textContent = moleculeSelected ? "Molecule" : "Crystal";
  button.dataset.kind = moleculeSelected ? "crystal" : "molecule";
  button.title = moleculeSelected ? "Switch to Crystal" : "Switch to Molecule";
}

function setStructureKind(kind) {
  if (importInProgress) {
    document.getElementById("status").textContent = "Loading in progress. Wait for it to finish.";
    return;
  }
  selectedStructureKind = kind === "molecule" ? "molecule" : "crystal";
  if (!structureLoadedForSelectedKind() && activeMode === "custom") {
    activeMode = "standard";
    invalidateCustomSequenceWork();
    syncActiveModeControls();
  }
  syncSourceKindControls();
  renderExampleOptions();
  renderStructureInfo();
  renderStatus();
}

function syncImportControls() {
  for (const id of ["import-cif", "cif-file", "import-molecule", "molecule-file", "example-select", "open-example"]) {
    document.getElementById(id).disabled = importInProgress;
  }
  for (const button of document.querySelectorAll(".structure-kind-button")) {
    button.disabled = importInProgress;
  }
  document.getElementById("example-select-display").disabled = importInProgress;
}

function renderExampleOptions() {
  const select = document.getElementById("example-select");
  const menu = document.getElementById("example-select-menu");
  const items = sortedExampleItems(exampleCatalog[selectedStructureKind] || [], selectedStructureKind);
  select.innerHTML = "";
  menu.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select example...";
  select.appendChild(placeholder);
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = exampleOptionText(item);
    if (item.error) option.title = item.error;
    select.appendChild(option);
    const menuItem = document.createElement("button");
    menuItem.type = "button";
    menuItem.className = "rich-select-option";
    menuItem.dataset.value = item.path;
    const prefix = document.createElement("span");
    prefix.textContent = `${item.formula ? `${item.formula} ` : ""}${item.name}`;
    menuItem.appendChild(prefix);
    if (item.symmetry) {
      menuItem.appendChild(document.createTextNode(" — "));
      const symmetry = document.createElement("span");
      symmetry.appendChild(renderHtml(formatGroupSymbol(item.symmetry)));
      menuItem.appendChild(symmetry);
    }
    menuItem.addEventListener("click", () => {
      select.value = item.path;
      selectedExamplePath = item.path;
      syncExamplePicker(items);
      setRichSelectOpen("example-picker", false);
    });
    menu.appendChild(menuItem);
  }
  const hasSelectedExample = items.some(item => item.path === selectedExamplePath);
  select.value = hasSelectedExample ? selectedExamplePath : "";
  syncExamplePicker(items);
  document.getElementById("open-example").disabled = importInProgress || items.length === 0;
}

function syncExamplePicker(items = sortedExampleItems(exampleCatalog[selectedStructureKind] || [], selectedStructureKind)) {
  const select = document.getElementById("example-select");
  const selected = items.find(item => item.path === select.value);
  const trigger = document.getElementById("example-select-display");
  trigger.replaceChildren();
  if (!selected) {
    trigger.textContent = "Select example...";
  } else {
    const prefix = document.createElement("span");
    prefix.textContent = `${exampleFormula(selected) ? `${exampleFormula(selected)} ` : ""}${selected.name}`;
    trigger.appendChild(prefix);
    if (selected.symmetry) {
      trigger.appendChild(document.createTextNode(" — "));
      trigger.appendChild(renderHtml(formatGroupSymbol(selected.symmetry)));
    }
  }
  for (const item of document.querySelectorAll("#example-select-menu .rich-select-option")) {
    item.classList.toggle("selected", item.dataset.value === select.value);
  }
}

function sortedExampleItems(items, kind) {
  const sorted = [...items];
  if (kind !== "crystal") return sorted;
  return sorted.sort((a, b) => {
    const numberA = spaceGroupNumber(a.symmetry);
    const numberB = spaceGroupNumber(b.symmetry);
    return compareNumber(numberA, numberB) || compareText(a.name || "", b.name || "");
  });
}

function spaceGroupNumber(label) {
  const match = String(label || "").match(/(?:No[.][ ]*)?([0-9]+)/);
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
}

function exampleFormula(item) {
  // display_formula keeps molecules readable (C6H6 rather than the reduced HC).
  return item.display_formula || item.formula || "";
}

function exampleOptionText(item) {
  const formula = exampleFormula(item) ? `${exampleFormula(item)} ` : "";
  const symmetry = item.symmetry ? formatPlainOverbar(formatSymbol(item.symmetry)) : "";
  return symmetry ? `${formula}${item.name} — ${symmetry}` : `${formula}${item.name}`;
}

function beginImport(message) {
  importInProgress = true;
  // Take the id from the clock, not from a counter. puzzle.js stamps its own
  // loads with Date.now() and the server keeps max(stored, incoming) so that
  // the newest load wins; a counter starting at 1 is below every timestamp the
  // puzzle has sent, so after playing a quiz every analysis-mode load came back
  // {"ok": false, "stale": true} and was silently discarded — the old structure
  // simply stayed on screen. Stay monotonic so two loads in the same
  // millisecond still get distinct ids.
  const requestId = Math.max(Date.now(), importRequestId + 1);
  importRequestId = requestId;
  hideLoadError();
  syncImportControls();
  document.getElementById("status").textContent = message;
  return requestId;
}

function finishImport(requestId) {
  if (requestId !== importRequestId) return;
  importInProgress = false;
  syncImportControls();
}

function isCurrentImport(requestId) {
  return requestId === importRequestId;
}

function hideLoadError() {
  const panel = document.getElementById("load-error-panel");
  if (panel) panel.hidden = true;
}

function showLoadError(title, rawError, context = {}) {
  const panel = document.getElementById("load-error-panel");
  if (!panel) return;
  const errorText = String(rawError || "Unknown error");
  const reason = predictedLoadFailureReason(errorText, context);
  document.getElementById("load-error-title").textContent = title;
  document.getElementById("load-error-message").textContent = reason;
  document.getElementById("load-error-detail").textContent = compactErrorDetail(errorText);
  panel.hidden = false;
  document.getElementById("status").textContent = `${title}: ${reason}`;
}

function predictedLoadFailureReason(errorText, context = {}) {
  const lower = String(errorText || "").toLowerCase();
  const name = context.name ? ` (${context.name})` : "";
  if (lower.includes("empty") && (lower.includes("cif") || lower.includes("file"))) {
    return `The selected file appears to be empty${name}.`;
  }
  if (lower.includes("analysis timed out")) {
    return `Analysis did not finish within the timeout${name}. The file may be too large, malformed, or triggering a slow symmetry analysis path.`;
  }
  if (lower.includes("already represented as a primitive cell")) {
    return `This structure is already represented as a primitive cell${name}.`;
  }
  if (lower.includes("already represented as a bravais cell")) {
    return `This structure is already represented as a Bravais cell${name}.`;
  }
  if (lower.includes("invalid json")) {
    return `The browser request was malformed before the file could be analyzed${name}. Try reopening the file.`;
  }
  if (lower.includes("zerodivisionerror") && lower.includes("pymatgen") && lower.includes("cif")) {
    return `The CIF parser failed before symmetry analysis${name}. A likely cause is malformed CIF loop syntax, such as a loop header with no data rows.`;
  }
  if (lower.includes("cif") && (lower.includes("parser") || lower.includes("parse") || lower.includes("from_file"))) {
    return `The file could not be parsed as a valid CIF${name}. Check CIF loop sections, quoted multiline fields, cell parameters, and atom-site columns.`;
  }
  if (lower.includes("no such file") || lower.includes("not found")) {
    return `The selected file could not be found${name}.`;
  }
  if (lower.includes("unsupported input file type")) {
    return `The selected file type is not supported${name}. Use CIF for crystals or XYZ for molecules.`;
  }
  return `The structure could not be loaded${name}. See details below for the parser or analysis error.`;
}

function compactErrorDetail(errorText) {
  const lines = String(errorText || "").split("\\n").map(line => line.trimEnd()).filter(Boolean);
  if (lines.length <= 14) return lines.join("\\n");
  return [
    ...lines.slice(0, 8),
    "...",
    ...lines.slice(-5),
  ].join("\\n");
}

function syncOperationSelection() {
  for (const row of document.querySelectorAll(".operation-row")) {
    const selected = Number(row.dataset.index) === state.operation_index;
    row.classList.toggle("selected", selected);
    row.setAttribute("aria-selected", selected ? "true" : "false");
  }
}

function fmtMatVal(v) {
  if (v === null || v === undefined) return "  ?";
  const r = Math.round(v * 1e9) / 1e9;
  if (Number.isInteger(r) && Math.abs(r) < 10) return String(r).padStart(3);
  return r.toFixed(4).padStart(8);
}

function fmtFrac(v) {
  if (v === null || v === undefined) return "?";
  const r = Math.round(v * 1e9) / 1e9;
  if (Math.abs(r) < 1e-8 || Math.abs(r - 1.0) < 1e-8) return "0";
  // try to express as a simple fraction with denominator ≤ 24
  for (const d of [2, 3, 4, 6, 8, 12, 24]) {
    const n = Math.round(r * d);
    if (Math.abs(n / d - r) < 1e-8) return n === 0 ? "0" : `${n}/${d}`;
  }
  return r.toFixed(4);
}

function appendOperationDetailLine(root, label, value, html = false) {
  const row = document.createElement("div");
  row.className = "operation-detail-line";
  const key = document.createElement("span");
  key.className = "operation-detail-label";
  key.textContent = label;
  const content = document.createElement("span");
  content.className = "operation-detail-value";
  if (html) content.appendChild(renderHtml(String(value)));
  else content.textContent = String(value);
  row.append(key, content);
  root.appendChild(row);
}

function appendOperationMatrix(root, label, matrix, formatter) {
  const block = document.createElement("div");
  block.className = "operation-matrix-block";
  const title = document.createElement("div");
  title.className = "operation-detail-label";
  title.textContent = label;
  const body = document.createElement("div");
  body.className = "operation-matrix";
  for (const values of matrix) {
    const row = document.createElement("div");
    row.className = "operation-matrix-row";
    for (const value of values) {
      const cell = document.createElement("span");
      cell.textContent = formatter(value).trim();
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
  block.append(title, body);
  root.appendChild(block);
}

function appendTransformCard(root, label, matrix, translation, formatter, translationFormatter) {
  const card = document.createElement("section");
  card.className = "operation-transform-card";
  const title = document.createElement("h3");
  title.textContent = label;
  card.appendChild(title);
  appendOperationMatrix(card, "W", matrix, formatter);
  appendOperationDetailLine(card, "t", translationFormatter(translation));
  root.appendChild(card);
}

function appendOperationTitle(root, value) {
  const title = document.createElement("div");
  title.className = "operation-detail-title";
  title.appendChild(renderHtml(value));
  root.appendChild(title);
}

function renderOperationDetails() {
  const div = document.getElementById("op-details");
  if (activeMode === "custom") {
    // Driven by copMatrix and its nested result rather than by `operations`,
    // so it has different inputs and keeps its own unconditional rebuild.
    lastOperationDetailsSignature = "";
    renderCustomOperationDetails();
    return;
  }
  const op = operations.find(o => o.index === state.operation_index);
  // Called from the 1 s poll and from every postState, almost always with
  // nothing changed. The summaries (element_summary / itc_like_summary) arrive
  // on a later fetch than the operation itself, so they belong in the
  // signature: without them the panel would keep the pre-summary text.
  const signature = JSON.stringify({
    source: state.json_path || "",
    reload: state.reload_request_id || 0,
    sourceKind,
    experienceMode,
    operationLabelMode,
    improperMode: state.improper_mode || "auto",
    selected: state.operation_index,
    operation: op && [
      op.index,
      op.symbol,
      op.display_symbol,
      op.element_summary,
      op.itc_like_summary,
    ],
  });
  if (signature === lastOperationDetailsSignature && div.childElementCount) return;
  lastOperationDetailsSignature = signature;
  div.replaceChildren();
  if (!op) return;
  appendOperationTitle(div, optionText(op));
  const transforms = document.createElement("div");
  transforms.className = "operation-transform-grid";
  div.appendChild(transforms);
  if (sourceKind === "molecule") {
    const Wc = op.matrix_cart;
    const tc = op.translation_cart;
    if (Wc) appendTransformCard(
      transforms, "Operation matrix", Wc, tc || [0, 0, 0], value => value.toFixed(4),
      values => `(${values.map(value => value.toFixed(4)).join(",  ")}) Å`,
    );
    return;
  }
  const W = op.matrix_frac;
  const t = op.translation_frac;
  if (!W || !t) return;

  appendTransformCard(transforms, "Fractional (W | t)", W, t, fmtMatVal, values => `(${values.map(fmtFrac).join(",  ")})`);
  const Wc = op.matrix_cart;
  const tc = op.translation_cart;
  if (Wc && tc) {
    appendTransformCard(
      transforms, "Cartesian (W | t)", Wc, tc, value => value.toFixed(4),
      values => `(${values.map(value => value.toFixed(4)).join(",  ")}) Å`,
    );
  }
}

function renderCustomOperationDetails() {
  const div = document.getElementById("op-details");
  div.replaceChildren();
  if (!copMatrix) {
    div.textContent = "Check the operation sequence to display its matrix.";
    return;
  }
  const result = copMatrix.result || {};
  appendOperationTitle(div, `Custom ${copMatrix.op_type}`);
  const metadata = document.createElement("div");
  metadata.className = "operation-detail-metadata";
  appendOperationDetailLine(metadata, "Status", result.is_symmetry ? "Symmetry operation" : "Not a symmetry operation");
  if (result.total !== undefined) {
    appendOperationDetailLine(metadata, "Atom mapping", `${result.mapped_count}/${result.total} mapped · ${result.unmapped_count} unmapped`);
  }
  const transforms = document.createElement("div");
  transforms.className = "operation-transform-grid";
  appendTransformCard(
    transforms, "Fractional", copMatrix.W_frac, copMatrix.t_frac, fmtMatVal,
    values => `(${values.map(fmtFrac).join(",  ")})`,
  );
  if (result.matrix_cart && result.translation_cart) {
    appendTransformCard(
      transforms, "Cartesian", result.matrix_cart, result.translation_cart,
      value => Number(value).toFixed(4),
      values => `(${values.map(value => Number(value).toFixed(4)).join(",  ")}) Å`,
    );
  }
  div.append(metadata, transforms);
}

function renderAtomElementFilter() {
  const root = document.getElementById("atom-element-filter");
  const elements = [...new Set(atoms.map(atom => atom.element))].sort(compareText);
  root.innerHTML = "";
  const allButton = document.createElement("button");
  allButton.type = "button";
  allButton.className = `direction-chip${atomElementFilterValue ? "" : " selected"}`;
  allButton.textContent = "All";
  allButton.addEventListener("click", () => {
    atomElementFilterValue = "";
    renderAtomElementFilter();
    renderAtoms();
  });
  root.appendChild(allButton);
  for (const element of elements) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `direction-chip${element === atomElementFilterValue ? " selected" : ""}`;
    button.textContent = element;
    button.addEventListener("click", () => {
      atomElementFilterValue = element;
      renderAtomElementFilter();
      renderAtoms();
    });
    root.appendChild(button);
  }
}

function renderElementColorControls() {
  const root = document.getElementById("element-colors");
  const elements = [...new Set(atoms.map(atom => atom.element))].sort(compareText);
  const elementColors = state.element_colors || {};
  const elementHidden = state.element_hidden || {};
  root.innerHTML = "";
  for (const element of elements) {
    const sample = atoms.find(atom => atom.element === element);
    const label = document.createElement("div");
    label.className = "element-color-item";
    const visible = !elementHidden[element];
    if (!visible) label.classList.add("hidden");
    const visibleButton = visibilityButton(visible, `${visible ? "Hide" : "Show"} ${element}`, () => {
      const next = Object.assign({}, state.element_hidden || {});
      if (visible) next[element] = true;
      else delete next[element];
      postState({element_hidden: next, playing: false, reset: true});
    });
    const input = document.createElement("input");
    input.type = "color";
    input.value = normalizeColor(elementColors[element] || (sample && sample.default_color));
    input.title = `Color for ${element}`;
    input.addEventListener("change", () => {
      const next = Object.assign({}, state.element_colors || {});
      next[element] = input.value;
      postState({element_colors: next});
    });
    const span = document.createElement("span");
    span.textContent = element;
    label.appendChild(visibleButton);
    label.appendChild(input);
    label.appendChild(span);
    root.appendChild(label);
  }
}

function renderAtoms() {
  const signature = atomListRenderSignature();
  if (signature === lastAtomRenderSignature) return;
  lastAtomRenderSignature = signature;
  const root = document.getElementById("atoms");
  const selected = new Set(state.selected_atoms || []);
  root.innerHTML = "";
  for (const atom of atoms) {
    if (atomElementFilterValue && atom.element !== atomElementFilterValue) continue;
    const label = document.createElement("label");
    label.className = "atom-row";
    if (customUnmappedAtoms.has(atom.index)) label.classList.add("unmapped");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "animation-toggle";
    checkbox.value = atom.index;
    checkbox.checked = selected.has(atom.index);
    checkbox.title = `Animate atom ${atom.index}`;
    checkbox.setAttribute("aria-label", `Animate atom ${atom.index}`);
    checkbox.addEventListener("change", onAtomSelectionChange);
    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = atomEffectiveColor(atom);
    colorInput.title = `Color for atom ${atom.index}`;
    colorInput.addEventListener("change", () => {
      const next = Object.assign({}, state.atom_colors || {});
      next[String(atom.index)] = colorInput.value;
      postState({atom_colors: next});
    });
    const visible = atomVisible(atom);
    const visibleButton = visibilityButton(visible, `${visible ? "Hide" : "Show"} atom ${atom.index}`, () => {
      postState(atomVisibilityUpdate(atom, visible));
    });
    const atomLabel = document.createElement("span");
    atomLabel.className = "atom-label";
    const atomIndex = document.createElement("span");
    atomIndex.className = "atom-index";
    atomIndex.textContent = `${atom.index}:`;
    atomLabel.appendChild(atomIndex);
    atomLabel.appendChild(document.createTextNode(atomDisplayLabel(atom)));
    if (customUnmappedAtoms.has(atom.index)) {
      const badge = document.createElement("span");
      badge.className = "atom-badge";
      badge.textContent = "do not map";
      atomLabel.appendChild(badge);
    }
    const motion = atomMotionParts(atom);
    const motionSpan = document.createElement("span");
    motionSpan.className = "atom-motion";
    if (motion) {
      const pathSpan = document.createElement("span");
      pathSpan.className = "atom-motion-path";
      pathSpan.textContent = motion.path;
      pathSpan.title = motion.path;
      const targetSpan = document.createElement("span");
      targetSpan.className = "atom-motion-target";
      targetSpan.textContent = `#${motion.target}`;
      motionSpan.appendChild(pathSpan);
      motionSpan.appendChild(targetSpan);
    }
    label.appendChild(checkbox);
    label.appendChild(atomLabel);
    label.appendChild(visibleButton);
    label.appendChild(colorInput);
    label.appendChild(motionSpan);
    root.appendChild(label);
  }
}

function renderSelectedAtomSummary() {
  const root = document.getElementById("selected-atom-summary");
  if (!root) return;
  root.innerHTML = "";
  if (experienceMode !== "beginner") return;
  const selected = String(state.scope).startsWith("selected")
    ? new Set((state.selected_atoms || []).map(Number))
    : new Set();
  const selectors = document.createElement("div");
  selectors.className = "atom-element-selectors";
  const selectionGroups = new Map();
  for (const atom of atoms) {
    const label = sourceKind === "crystal" ? atomDisplayLabel(atom) : atom.element;
    if (!selectionGroups.has(label)) selectionGroups.set(label, []);
    selectionGroups.get(label).push(Number(atom.index));
  }
  for (const [label, indices] of [...selectionGroups.entries()].sort((a, b) => compareText(a[0], b[0]))) {
    const allSelected = indices.length > 0 && indices.every(index => selected.has(index));
    const button = document.createElement("button");
    button.type = "button";
    button.className = `atom-element-selector${allSelected ? " selected" : ""}`;
    button.textContent = label;
    button.addEventListener("click", event => {
      let next;
      if (event.ctrlKey || event.metaKey || event.shiftKey) {
        next = new Set(selected);
        for (const index of indices) {
          if (allSelected) next.delete(index);
          else next.add(index);
        }
      } else {
        next = allSelected && selected.size === indices.length ? new Set() : new Set(indices);
      }
      postState({
        selected_atoms: [...next].sort((a, b) => a - b),
        scope: next.size ? "selected_displayed" : "displayed",
        playing: false,
        reset: true,
      });
    });
    selectors.appendChild(button);
  }
  const resetAppearance = document.createElement("button");
  resetAppearance.type = "button";
  resetAppearance.className = "atom-element-selector reset-appearance-button";
  resetAppearance.textContent = "Reset appearance";
  resetAppearance.addEventListener("click", resetAtomAppearance);
  selectors.appendChild(resetAppearance);
  root.appendChild(selectors);
  if (!selected.size) return;
  const details = document.createElement("details");
  details.className = "selected-atom-details";
  details.open = selectedAtomDetailsOpen;
  details.addEventListener("toggle", () => {
    selectedAtomDetailsOpen = details.open;
  });
  const summary = document.createElement("summary");
  summary.textContent = `Selected atoms (${selected.size})`;
  details.appendChild(summary);
  const cards = document.createElement("div");
  cards.className = "selected-atom-cards";
  for (const atom of atoms.filter(item => selected.has(Number(item.index)))) {
    const motion = atomMotionBySource.get(atom.index);
    const start = sourceKind === "crystal" ? motion?.start_frac : motion?.start_cart;
    const target = sourceKind === "crystal" ? motion?.target_frac : motion?.target_cart;
    const formatter = sourceKind === "crystal" ? formatFrac : formatCoord;
    const unit = sourceKind === "crystal"
      ? (experienceMode === "advanced" ? " fractional" : "")
      : " Å";
    const card = document.createElement("div");
    card.className = "selected-atom-card";
    const header = document.createElement("div");
    header.className = "selected-atom-card-header";
    const name = document.createElement("div");
    name.className = "selected-atom-name";
    name.textContent = `${atomDisplayLabel(atom)}（atom ${atom.index}）`;
    const controls = document.createElement("div");
    controls.className = "selected-atom-controls";
    const visible = atomVisible(atom);
    const visibleButton = visibilityButton(visible, `${visible ? "Hide" : "Show"} atom ${atom.index}`, () => {
      postState(atomVisibilityUpdate(atom, visible));
    });
    const colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.value = atomEffectiveColor(atom);
    colorInput.title = `Color for atom ${atom.index}`;
    colorInput.setAttribute("aria-label", `Color for atom ${atom.index}`);
    colorInput.addEventListener("change", () => {
      const next = Object.assign({}, state.atom_colors || {});
      next[String(atom.index)] = colorInput.value;
      postState({atom_colors: next});
    });
    controls.append(visibleButton, colorInput);
    header.append(name, controls);
    const coordinates = document.createElement("div");
    coordinates.className = "selected-atom-coordinates";
    coordinates.textContent = Array.isArray(start) && Array.isArray(target)
      ? `${formatVector(start, formatter)} → ${formatVector(target, formatter)}${unit}`
      : "座標情報を取得できません";
    card.appendChild(header);
    card.appendChild(coordinates);
    cards.appendChild(card);
  }
  details.appendChild(cards);
  root.appendChild(details);
}

function operationListRenderSignature(sorted) {
  return JSON.stringify({
    // The rows must be keyed on what they say, not just how many there are.
    // `operations` and `state` are refreshed by separate responses, so there is
    // a moment when the new structure's json_path is paired with the previous
    // structure's operations; benzene and methane both have 24 operations
    // numbered 0-23, and indices alone left methane's rows on screen.
    source: state.json_path || "",
    reload: state.reload_request_id || 0,
    operations: sorted.map(operation => [
      operation.index,
      operation.symbol || "",
      operation.display_symbol || "",
    ]),
    selected: state.operation_index,
    operationTypeFilterValue,
    directionFilterValue,
    experienceMode,
    improperMode: state.improper_mode || "auto",
    // element_summary, the axis/plane/center group headers and the per-row
    // direction hint all read fields that only arrive once summaries are
    // ready. Without this, the first (pre-summary) render's signature still
    // matches once the real data lands, and the refresh that re-fetches
    // `operations` at that point (see refreshState) silently no-ops here.
    summariesReady,
  });
}

function atomListRenderSignature() {
  return JSON.stringify({
    atoms: atoms.map(atom => [
      atom.index,
      atom.element,
      atom.asymmetric_index,
      atom.default_color,
    ]),
    selected_atoms: state.selected_atoms || [],
    element_colors: state.element_colors || {},
    atom_colors: state.atom_colors || {},
    element_hidden: state.element_hidden || {},
    atom_hidden: state.atom_hidden || {},
    atomElementFilterValue,
    unmapped: [...customUnmappedAtoms].sort((a, b) => a - b),
    motion: [...atomMotionBySource.entries()].map(([source, motion]) => [
      source,
      motion.target_atom,
      motion.start_frac,
      motion.target_frac,
      motion.start_cart,
      motion.target_cart,
    ]),
  });
}

function syncSpeedButtons() {
  const speed = Number(state.speed || 1.0);
  const slider = document.getElementById("animation-speed");
  const output = document.getElementById("animation-speed-value");
  if (document.activeElement !== slider) slider.value = String(speed);
  output.value = `${speed.toFixed(2)}×`;
  output.textContent = output.value;
  document.getElementById("pause-at-breakpoints").checked = Boolean(state.pause_at_breakpoints);
  document.getElementById("show-trajectories").checked = Boolean(state.show_trajectories);
}

function syncBoundaryButtons() {
  const boundaryMode = state.animation_boundary_mode || "continuous";
  for (const button of document.querySelectorAll(".boundary-button")) {
    button.classList.toggle("selected", button.dataset.boundaryMode === boundaryMode);
  }
}

const displayRangeSteps = [
  {mode: "source", label: "Unit cell"},
  {mode: "expanded_quarter", label: "±1/4"},
  {mode: "expanded_half", label: "±1/2"},
  {mode: "expanded_0_75", label: "±3/4"},
  {mode: "expanded_1_0", label: "±1"},
];

function syncDisplayButtons() {
  const displayMode = state.display_mode || "source";
  const index = Math.max(0, displayRangeSteps.findIndex(step => step.mode === displayMode));
  document.getElementById("display-range-value").textContent = displayRangeSteps[index].label;
  document.getElementById("display-range-decrease").disabled = index === 0;
  document.getElementById("display-range-increase").disabled = index === displayRangeSteps.length - 1;
  document.getElementById("include-boundary-images").checked = Boolean(state.include_boundary_images);
}

async function changeDisplayRange(offset) {
  if (displayRangeUpdatePending) return;
  const displayMode = state.display_mode || "source";
  const currentIndex = Math.max(0, displayRangeSteps.findIndex(step => step.mode === displayMode));
  const nextIndex = Math.max(0, Math.min(displayRangeSteps.length - 1, currentIndex + offset));
  if (nextIndex === currentIndex) return;
  displayRangeUpdatePending = true;
  state.display_mode = displayRangeSteps[nextIndex].mode;
  syncDisplayButtons();
  document.getElementById("display-range-decrease").disabled = true;
  document.getElementById("display-range-increase").disabled = true;
  try {
    await postState({
      display_mode: displayRangeSteps[nextIndex].mode,
      playing: false,
      reset: true,
    });
  } catch (error) {
    state.display_mode = displayMode;
    throw error;
  } finally {
    displayRangeUpdatePending = false;
    syncDisplayButtons();
  }
}

function syncCellOriginButtons() {
  const cellOriginMode = state.cell_origin_mode || "center";
  for (const button of document.querySelectorAll(".cell-origin-button")) {
    button.classList.toggle("selected", button.dataset.cellOriginMode === cellOriginMode);
  }
}

function syncCellSettingButtons() {
  const cellSettingMode = state.cell_setting_mode || "native";
  for (const button of document.querySelectorAll(".cell-setting-button")) {
    button.classList.toggle("selected", button.dataset.cellSettingMode === cellSettingMode);
  }
}

function setCellSettingBusy(busy, mode = "") {
  for (const button of document.querySelectorAll(".cell-setting-button")) {
    button.disabled = busy;
  }
  if (busy) {
    document.getElementById("status").textContent =
      `Converting cell setting to ${formatCellSettingLabel(mode)}...`;
  }
}

function syncProjectionButtons() {
  const perspective = (state.projection_mode || "perspective") === "perspective";
  const button = document.getElementById("projection-toggle");
  if (!button) return;
  button.textContent = perspective ? "Perspective" : "Orthographic";
  button.dataset.projectionMode = perspective ? "orthographic" : "perspective";
  button.title = perspective ? "Switch to orthographic" : "Switch to perspective";
}

function syncBackgroundButtons() {
  const dark = (state.background_mode || "light") === "dark";
  const button = document.getElementById("background-toggle");
  if (!button) return;
  button.textContent = dark ? "Black" : "White";
  button.dataset.backgroundMode = dark ? "light" : "dark";
  button.title = dark ? "Switch to white background" : "Switch to black background";
}

function syncLegendButtons() {
  const visible = Boolean(state.legend_visible);
  const button = document.getElementById("legend-toggle");
  if (!button) return;
  button.textContent = visible ? "Legend on" : "Legend off";
  button.dataset.legendVisible = visible ? "false" : "true";
  button.title = visible ? "Hide legend" : "Show legend";
}

function syncImproperModeControl() {
  document.getElementById("improper-mode").value = state.improper_mode || "auto";
}

function syncAtomModeButtons() {
  const scope = state.scope || "representative";
  for (const button of document.querySelectorAll(".atom-mode-button")) {
    button.classList.toggle("selected", button.dataset.scope === scope);
  }
}

function syncPlayToggleButton() {
  const button = document.getElementById("play-toggle");
  const boundaryPaused = atPausedAnimationBoundary();
  button.textContent = boundaryPaused ? "次へ" : state.playing ? "停止" : "再生";
  button.classList.toggle("secondary", Boolean(state.playing) || boundaryPaused);
  button.disabled = false;
}

function setMovementProgress(progress) {
  const value = Math.max(0, Math.min(Number(progress) || 0, 1));
  movementProgressValue = value;
  const slider = document.getElementById("movement-progress");
  const output = document.getElementById("movement-progress-value");
  const previous = document.getElementById("previous-frame");
  const next = document.getElementById("next-frame");
  if (slider) slider.value = String(Math.round(value * 1000));
  if (output) output.textContent = `${Math.round(value * 100)}%`;
  if (previous) previous.disabled = value <= 0;
  if (next) next.disabled = value >= 1;
  syncPlayToggleButton();
}

function setMovementBreakpoints(values) {
  movementBreakpoints = [...new Set((values || [0, 1]).map(Number).filter(Number.isFinite))]
    .filter(value => value >= 0 && value <= 1)
    .sort((a, b) => a - b);
  const list = document.getElementById("movement-stops");
  if (!list) return;
  list.innerHTML = "";
  for (const value of movementBreakpoints) {
    const option = document.createElement("option");
    option.value = String(Math.round(value * 1000));
    list.appendChild(option);
  }
}

function dispatchMovementProgress(progress) {
  setMovementProgress(progress);
  window.dispatchEvent(new CustomEvent("symmetry-animation-progress", {detail: {progress}}));
}

function stepMovementProgress(direction) {
  const current = movementProgressValue;
  const delta = Math.sign(direction) * MOVEMENT_FRAME_STEP;
  const proposed = Math.max(0, Math.min(current + delta, 1));
  const crossedBreakpoints = movementBreakpoints.filter(value => (
    delta > 0
      ? value > current + 1e-9 && value < proposed - 1e-9
      : value < current - 1e-9 && value > proposed + 1e-9
  ));
  const progress = crossedBreakpoints.length
    ? (delta > 0 ? Math.min(...crossedBreakpoints) : Math.max(...crossedBreakpoints))
    : proposed;
  if (state.playing) postState({playing: false});
  dispatchMovementProgress(progress);
}

function selectedAtomIndices() {
  return Array.from(document.querySelectorAll("#atoms input.animation-toggle:checked"))
    .map(input => Number(input.value));
}

function onAtomSelectionChange() {
  const selected = selectedAtomIndices();
  postState({
    selected_atoms: selected,
    scope: selected.length > 0 ? "selected" : "displayed",
    playing: false,
    reset: true,
  });
}

function renderStatus() {
  if (!structureLoadedForSelectedKind()) {
    const config = sourceKindConfig(selectedStructureKind);
    document.getElementById("status").textContent =
      `No ${selectedStructureKind} loaded. Open an example or ${config.inputLabel}.\\n` +
      `import: ${state.import_status || "-"}`;
    return;
  }
  const operation = operations.find(op => op.index === state.operation_index);
  const selected = state.selected_atoms && state.selected_atoms.length
    ? state.selected_atoms.join(", ")
    : "-";
  const scopeLabel = (state.scope || "representative") === "representative"
    ? "clear"
    : (state.scope || "representative");
  document.getElementById("status").textContent =
    `state: ${state.playing ? "playing" : "stopped"}\\n` +
    `mode: ${activeMode}\\n` +
    `operation: ${activeMode === "custom" ? (copMatrix ? "custom " + copMatrix.op_type : "custom unchecked") : (operation ? stripHtml(optionText(operation)) : state.operation_index)}\\n` +
    `speed: ${state.speed || 1.0}x\\n` +
    `boundary: ${state.animation_boundary_mode || "continuous"}\\n` +
    `projection: ${state.projection_mode || "perspective"}\\n` +
    `cell: ${state.cell_setting_mode || "native"}\\n` +
    `display: ${state.display_mode || "source"}\\n` +
    `scope: ${scopeLabel}\\n` +
    `selected atoms: ${selected}\\n` +
    `source: ${state.json_path || "-"}\\n` +
    `import: ${state.import_status || "-"}\\n` +
    `status: ${state.status_message || "-"}`;
}

function viewCenterValue(id, fallback = 0) {
  const value = Number(document.getElementById(id).value);
  return Number.isFinite(value) ? value : fallback;
}

function viewDirectionValue(id, fallback = 0) {
  const value = Number(document.getElementById(id).value);
  return Number.isFinite(value) ? value : fallback;
}

function applyViewCenter() {
  postState({
    view_center_request_id: Date.now(),
    view_center_frac: [
      viewCenterValue("view-center-x"),
      viewCenterValue("view-center-y"),
      viewCenterValue("view-center-z"),
    ],
  });
}

function applyViewDirectionIndex() {
  postState({
    view_direction_request_id: Date.now(),
    view_direction_frac: [
      viewDirectionValue("view-dir-h"),
      viewDirectionValue("view-dir-k"),
      viewDirectionValue("view-dir-l", 1),
    ],
    playing: false,
  });
}

function applyViewPlaneIndex() {
  postState({
    view_plane_request_id: Date.now(),
    view_plane_hkl: [
      viewDirectionValue("view-plane-h"),
      viewDirectionValue("view-plane-k"),
      viewDirectionValue("view-plane-l", 1),
    ],
    playing: false,
  });
}

function syncSourceKindControls() {
  sourceKind = state.source_kind || "crystal";
  syncStructureKindButtons();
  const loadedForSelected = structureLoadedForSelectedKind();
  document.querySelector(".viewer-workspace").hidden = !loadedForSelected;
  document.getElementById("workspace").hidden = !loadedForSelected;
  document.getElementById("utility-panel").hidden = !loadedForSelected;
  document.getElementById("start-panel").hidden = loadedForSelected;
  document.getElementById("import-cif").hidden = selectedStructureKind !== "crystal";
  document.getElementById("import-molecule").hidden = selectedStructureKind !== "molecule";
  renderExampleOptions();
  document.getElementById("display-block").hidden = !loadedForSelected || sourceKind !== "crystal";
  document.getElementById("boundary-images-toggle").hidden = !loadedForSelected || sourceKind !== "crystal";
  document.getElementById("view-direction-index-block").hidden = !loadedForSelected || sourceKind !== "crystal";
  document.getElementById("view-plane-index-block").hidden = !loadedForSelected || sourceKind !== "crystal";
  document.getElementById("unit-cell-atoms").hidden = sourceKind !== "crystal";
  syncOperationLabelModeControls();
  const config = sourceKindConfig(sourceKind);
  document.querySelector("#standard-panel .section-title").textContent = "Operations";
  if ((!loadedForSelected || sourceKind !== "crystal") && activeMode === "custom") {
    activeMode = "standard";
    invalidateCustomSequenceWork();
    syncActiveModeControls();
  }
  const label = document.getElementById("view-center-label");
  if (sourceKind !== "crystal") {
    label.textContent = config.viewCenterLabel;
    for (const id of ["view-center-x", "view-center-y", "view-center-z"]) {
      document.getElementById(id).value = "0";
    }
  } else {
    label.textContent = config.viewCenterLabel;
  }
}

async function postState(update) {
  if (loopRestartTimer !== null && update.reset && !loopRestartInProgress) cancelLoopRestart();
  state = await api("/api/state", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(update),
  });
  window.dispatchEvent(new CustomEvent("symmetry-state-update", {
    detail: {state: {...state}, update: {...update}},
  }));
  await refreshAtomMotion();
  syncSpeedButtons();
  syncBoundaryButtons();
  syncDisplayButtons();
  syncCellOriginButtons();
  syncCellSettingButtons();
  syncProjectionButtons();
  syncBackgroundButtons();
  syncLegendButtons();
  syncImproperModeControl();
  syncOperationLabelModeControls();
  syncAtomModeButtons();
  syncPlayToggleButton();
  renderElementColorControls();
  renderDirectionFilter();
  renderOperations();
  renderAtoms();
  renderSelectedAtomSummary();
  renderStructureInfo();
  renderStatus();
  renderOperationDetails();
}

window.symmetryPostState = postState;

async function applyCellSetting(mode) {
  setCellSettingBusy(true, mode);
  try {
    const result = await api("/api/cell_setting", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({cell_setting_mode: mode}),
    });
    hideLoadError();
    operations = result.operations || [];
    atoms = result.atoms || [];
    state = result.state || {};
    directionFilterValue = "";
    atomElementFilterValue = "";
    summariesReady = Boolean(state.summaries_ready);
    customUnmappedAtoms = new Set();
    customOperationSequence = [];
    copMatrix = null;
    activeMode = "standard";
    document.getElementById("cop-result").hidden = true;
    invalidateCustomSequenceWork();
    syncActiveModeControls();
    syncSourceKindControls();
    renderDirectionFilter();
    renderAtomElementFilter();
    renderElementColorControls();
    syncOperationSelection();
    syncSpeedButtons();
    syncBoundaryButtons();
    syncDisplayButtons();
    syncCellOriginButtons();
    syncCellSettingButtons();
    syncProjectionButtons();
    syncBackgroundButtons();
    syncLegendButtons();
    syncImproperModeControl();
    syncAtomModeButtons();
    syncPlayToggleButton();
      renderOperations();
    await refreshAtomMotion();
    renderAtoms();
    renderSelectedAtomSummary();
    renderStructureInfo();
    renderStatus();
    renderOperationDetails();
  } finally {
    setCellSettingBusy(false);
  }
}

async function refreshAtomMotion() {
  if (!structureLoadedForSelectedKind() || activeMode !== "standard") {
    atomMotionBySource = new Map();
    lastAtomMotionRequestKey = "";
    pendingAtomMotionRequestKey = "";
    atomMotionRequestGeneration += 1;
    return;
  }
  const requestKey = JSON.stringify([
    state.json_path,
    state.reload_request_id,
    state.cell_setting_mode,
    state.operation_index,
    state.improper_mode,
    state.display_mode,
    state.cell_origin_mode,
  ]);
  if (requestKey === lastAtomMotionRequestKey || requestKey === pendingAtomMotionRequestKey) return;
  pendingAtomMotionRequestKey = requestKey;
  const generation = ++atomMotionRequestGeneration;
  try {
    const result = await api("/api/atom_motion");
    if (generation !== atomMotionRequestGeneration) return;
    atomMotionBySource = new Map(
      (result.entries || [])
        .filter(entry => entry.source_atom !== null && entry.source_atom !== undefined)
        .map(entry => [entry.source_atom, entry])
    );
    lastAtomMotionRequestKey = requestKey;
  } catch (error) {
    if (generation !== atomMotionRequestGeneration) return;
    atomMotionBySource = new Map();
    lastAtomMotionRequestKey = "";
    console.warn("Atom motion refresh failed", error);
  } finally {
    if (generation === atomMotionRequestGeneration) pendingAtomMotionRequestKey = "";
  }
}

async function importCifFile() {
  if (importInProgress) {
    document.getElementById("status").textContent = "Loading in progress. Wait for it to finish.";
    return;
  }
  const input = document.getElementById("cif-file");
  const file = input.files && input.files[0];
  if (!file) {
    document.getElementById("status").textContent = "Choose a CIF file first.";
    return;
  }
  const requestId = beginImport(`Loading ${file.name}...`);
  try {
    const content = await file.text();
    const result = await api("/api/import_cif", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({filename: file.name, content, request_id: requestId}),
    });
    if (!isCurrentImport(requestId)) return;
    await applyLoadedStructureSafely(result, "CIF import failed", "CIF display failed", "", {name: file.name, kind: "CIF"});
  } finally {
    finishImport(requestId);
  }
}

async function importMoleculeFile() {
  if (importInProgress) {
    document.getElementById("status").textContent = "Loading in progress. Wait for it to finish.";
    return;
  }
  const input = document.getElementById("molecule-file");
  const file = input.files && input.files[0];
  if (!file) {
    document.getElementById("status").textContent = "Choose an XYZ file first.";
    return;
  }
  const requestId = beginImport(`Loading ${file.name}...`);
  try {
    const content = await file.text();
    const result = await api("/api/import_molecule", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({filename: file.name, content, request_id: requestId}),
    });
    if (!isCurrentImport(requestId)) return;
    await applyLoadedStructureSafely(result, "Molecule import failed", "Molecule display failed", "", {name: file.name, kind: "XYZ"});
  } finally {
    finishImport(requestId);
  }
}

async function openSelectedExample() {
  if (importInProgress) {
    document.getElementById("status").textContent = "Loading in progress. Wait for it to finish.";
    return;
  }
  const select = document.getElementById("example-select");
  const path = select.value;
  if (!path) {
    document.getElementById("status").textContent = "Choose an example first.";
    return;
  }
  const requestId = beginImport(`Loading ${path}...`);
  try {
    const result = await api("/api/open_example", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({kind: selectedStructureKind, path, request_id: requestId}),
    });
    if (!isCurrentImport(requestId)) return;
    await applyLoadedStructureSafely(result, "Example load failed", "Example display failed", path, {name: path, kind: selectedStructureKind});
  } finally {
    finishImport(requestId);
  }
}

async function applyLoadedStructureSafely(result, fallbackError, displayError, examplePath = "", context = {}) {
  try {
    await applyLoadedStructure(result, fallbackError, examplePath, context);
  } catch (error) {
    console.error(displayError, error);
    showLoadError(displayError, error, context);
  }
}

async function applyLoadedStructure(result, fallbackError, examplePath = "", context = {}) {
  if (result.stale) {
    state = result.state || state;
    renderStatus();
    return;
  }
  if (!result.ok) {
    state = result.state || state;
    renderStatus();
    showLoadError(fallbackError, result.error || fallbackError, context);
    return;
  }
  hideLoadError();
  operations = result.operations || [];
  atoms = result.atoms || [];
  state = result.state || {};
  if (state.source_kind === "crystal" || state.source_kind === "molecule") {
    selectedStructureKind = state.source_kind;
  }
  sourceKind = state.source_kind || "crystal";
  selectedExamplePath = examplePath || "";
  setDefaultOperationSortForSourceKind();
  directionFilterValue = "";
  atomElementFilterValue = "";
  summariesReady = Boolean(state.summaries_ready);
  customUnmappedAtoms = new Set();
  customOperationSequence = [];
  copMatrix = null;
  document.getElementById("cop-result").hidden = true;
  invalidateCustomSequenceWork();
  activeMode = "standard";
  syncActiveModeControls();
  syncSourceKindControls();
  renderDirectionFilter();
  renderAtomElementFilter();
  renderElementColorControls();
  syncOperationSelection();
  syncSpeedButtons();
  syncBoundaryButtons();
  syncDisplayButtons();
  syncCellOriginButtons();
  syncCellSettingButtons();
  syncProjectionButtons();
  syncBackgroundButtons();
  syncLegendButtons();
  syncImproperModeControl();
  syncAtomModeButtons();
  syncPlayToggleButton();
  renderOperations();
  await refreshAtomMotion();
  renderAtoms();
  renderSelectedAtomSummary();
  renderStructureInfo();
  renderStatus();
  renderOperationDetails();
}

function syncActiveModeControls() {
  const customAvailable = experienceMode === "advanced"
    && sourceKind === "crystal"
    && structureLoadedForSelectedKind();
  if (!customAvailable && activeMode === "custom") {
    activeMode = "standard";
    invalidateCustomSequenceWork();
  }
  const standard = activeMode !== "custom";
  document.getElementById("standard-panel").hidden = !standard;
  const customPanel = document.getElementById("custom-panel");
  customPanel.hidden = standard || !customAvailable;
  const modeControls = document.getElementById("mode-controls");
  modeControls.hidden = !customAvailable;
  for (const button of modeControls.querySelectorAll(".mode-button")) {
    const selected = button.dataset.mode === activeMode;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
  }
}

function syncExperienceModeControls() {
  const beginner = experienceMode === "beginner";
  document.body.classList.toggle("beginner-mode", beginner);
  document.body.classList.toggle("advanced-mode", !beginner);
  const modeButton = document.getElementById("experience-toggle");
  const currentMode = beginner ? "Simple" : "Full";
  const nextMode = beginner ? "Full" : "Simple";
  modeButton.textContent = `${currentMode} → ${nextMode}`;
  modeButton.dataset.experience = beginner ? "advanced" : "beginner";
  modeButton.title = `Currently ${currentMode}; switch to ${nextMode} mode`;
  modeButton.setAttribute("aria-label", `Currently ${currentMode} mode. Switch to ${nextMode} mode.`);
  if (beginner) {
    cancelLoopRestart();
    setLoopAnimationEnabled(false);
  }
  document.getElementById("main-title").textContent = "Symmetry Viewer";
  document.getElementById("operations-title").textContent = "Operations";
  document.getElementById("operations-label").textContent = "操作";
  document.getElementById("animation-title").textContent = "Animation";
  document.getElementById("view-title").textContent = "カメラ";
  document.getElementById("projection-label").textContent = "投影";
  document.getElementById("cell-title").textContent = "セル";
  document.getElementById("range-label").textContent = "表示範囲";
  syncPlayToggleButton();
}

function setExperienceMode(mode) {
  experienceMode = mode === "advanced" ? "advanced" : "beginner";
  if (experienceMode === "beginner") {
    directionFilterValue = "";
    const update = {};
    if (activeMode !== "standard") {
      activeMode = "standard";
      invalidateCustomSequenceWork();
      Object.assign(update, {active_mode: "standard", playing: false, reset: true, clear_custom_check: true});
    }
    if (state.pause_at_breakpoints) {
      state.pause_at_breakpoints = false;
      update.pause_at_breakpoints = false;
    }
    if (Object.keys(update).length) postState(update);
  }
  syncActiveModeControls();
  syncExperienceModeControls();
  renderOperations();
  renderStructureInfo();
  renderSelectedAtomSummary();
  renderOperationDetails();
}

async function setActiveMode(mode) {
  activeMode = mode === "custom" ? "custom" : "standard";
  syncActiveModeControls();
  if (activeMode === "standard") {
    customUnmappedAtoms = new Set();
    invalidateCustomSequenceWork();
    renderAtoms();
    await postState({active_mode: "standard", playing: false, reset: true, clear_custom_check: true});
  } else {
    renderCustomSequenceControls();
    await postState({active_mode: "custom", playing: false, reset: true});
  }
  renderStatus();
  renderOperationDetails();
}

for (const button of document.querySelectorAll(".mode-button")) {
  button.addEventListener("click", () => setActiveMode(button.dataset.mode));
}

async function refreshState() {
  if (refreshInProgress) return;
  // Nothing of the analysis panel is on screen during a quiz, so the 1 s poll
  // would fetch and re-sync a dozen controls nobody can see. Re-entering
  // analysis calls refreshAnalysisFromServer(), so no state is lost by
  // skipping. The 3D view's own loop already bails on the same class.
  if (document.body.classList.contains("in-puzzle")) return;
  refreshInProgress = true;
  try {
    state = await api("/api/state");
    syncOperationSelection();
    syncSpeedButtons();
    syncBoundaryButtons();
    syncDisplayButtons();
    syncCellOriginButtons();
    syncCellSettingButtons();
    syncProjectionButtons();
    syncBackgroundButtons();
    syncLegendButtons();
    syncImproperModeControl();
    syncAtomModeButtons();
    syncPlayToggleButton();
    renderStructureInfo();
    renderStatus();
    renderOperationDetails();
    if (!summariesReady && state.summaries_ready) {
      summariesReady = true;
      const info = await api("/api/operations");
      operations = info.operations;
      renderDirectionFilter();
      renderOperations();
      renderOperationDetails();
    }
  } finally {
    refreshInProgress = false;
  }
}

document.getElementById("operation-sort").addEventListener("change", () => {
  directionFilterValue = "";
  renderDirectionFilter();
  renderOperations();
});
document.getElementById("operation-label-mode").addEventListener("change", event => {
  operationLabelMode = event.target.value === "itc_like" ? "itc_like" : "standard";
  syncOperationLabelModeControls();
  renderOperations();
  renderOperationDetails();
  renderStatus();
});
for (const button of document.querySelectorAll(".structure-kind-button")) {
  button.addEventListener("click", () => setStructureKind(button.dataset.kind));
}
document.getElementById("import-cif").addEventListener("click", () => {
  const input = document.getElementById("cif-file");
  input.value = "";
  input.click();
});
document.getElementById("cif-file").addEventListener("change", () => {
  importCifFile().catch(error => {
    const file = document.getElementById("cif-file").files && document.getElementById("cif-file").files[0];
    showLoadError("CIF import failed", error, {name: file && file.name, kind: "CIF"});
  });
});
document.getElementById("import-molecule").addEventListener("click", () => {
  const input = document.getElementById("molecule-file");
  input.value = "";
  input.click();
});
document.getElementById("molecule-file").addEventListener("change", () => {
  importMoleculeFile().catch(error => {
    const file = document.getElementById("molecule-file").files && document.getElementById("molecule-file").files[0];
    showLoadError("Molecule import failed", error, {name: file && file.name, kind: "XYZ"});
  });
});
document.getElementById("open-example").addEventListener("click", () => {
  openSelectedExample().catch(error => {
    const select = document.getElementById("example-select");
    showLoadError("Example load failed", error, {name: select && select.value, kind: selectedStructureKind});
  });
});
for (const button of document.querySelectorAll(".experience-button")) {
  button.addEventListener("click", () => setExperienceMode(button.dataset.experience));
}
function startAnimation() {
  if (activeMode === "custom") {
    return sendCurrentCustomAnimation(true);
  }
  return postState({playing: true});
}

function loopPlaybackEnabled() {
  return experienceMode === "advanced" && document.getElementById("loop-animation").checked;
}

function cancelLoopRestart() {
  if (loopRestartTimer !== null) window.clearTimeout(loopRestartTimer);
  loopRestartTimer = null;
}

function setLoopAnimationEnabled(enabled) {
  document.getElementById("loop-animation").checked = enabled;
  if (!enabled) cancelLoopRestart();
}

function scheduleLoopRestart() {
  if (!loopPlaybackEnabled()) return;
  cancelLoopRestart();
  loopRestartTimer = window.setTimeout(async () => {
    loopRestartTimer = null;
    if (!loopPlaybackEnabled() || state.playing) return;
    loopRestartInProgress = true;
    try {
      if (activeMode === "custom") startAnimation();
      else await postState({playing: true, reset: true});
    } finally {
      loopRestartInProgress = false;
    }
  }, LOOP_RESTART_DELAY_MS);
}

window.addEventListener("symmetry-animation-complete", scheduleLoopRestart);

let pausePreferenceUpdate = Promise.resolve();
async function toggleAnimationPlayback() {
  await pausePreferenceUpdate;
  cancelLoopRestart();
  if (atPausedAnimationBoundary()) {
    await postState({playing: true});
    return;
  }
  if (state.playing) {
    await postState({playing: false});
  } else {
    await startAnimation();
  }
}

function resetAnimationPlayback() {
  cancelLoopRestart();
  return postState({playing: false, reset: true});
}

document.getElementById("play-toggle").addEventListener("click", toggleAnimationPlayback);
document.getElementById("reset").addEventListener("click", resetAnimationPlayback);
document.getElementById("loop-animation").addEventListener("change", event => {
  setLoopAnimationEnabled(event.currentTarget.checked);
});
document.getElementById("previous-frame").addEventListener("click", () => stepMovementProgress(-1));
function atPausedAnimationBoundary() {
  if (!state.pause_at_breakpoints || state.playing || movementProgressValue >= 1 - 1e-9) return false;
  return movementBreakpoints.some(value => (
    value > 1e-9 && value < 1 - 1e-9 && Math.abs(value - movementProgressValue) <= 0.0015
  ));
}

document.getElementById("next-frame").addEventListener("click", () => stepMovementProgress(1));
document.addEventListener("keydown", event => {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  const target = event.target;
  const editingControl = target instanceof HTMLElement && (
    target.isContentEditable
    || target.matches('input:not([type="range"]), select, textarea')
  );
  if (editingControl) return;
  event.preventDefault();
  stepMovementProgress(event.key === "ArrowLeft" ? -1 : 1);
});
const movementProgress = document.getElementById("movement-progress");

function stopAnimationForScrubbing() {
  if (state.playing) postState({playing: false});
}

function updateMovementProgressFromInput(input) {
  const progress = Number(input.value) / 1000;
  dispatchMovementProgress(progress);
}

function snapMovementProgressFromInput(input) {
  const progress = Number(input.value) / 1000;
  const nearest = movementBreakpoints.reduce(
    (best, value) => Math.abs(value - progress) < Math.abs(best - progress) ? value : best,
    movementBreakpoints[0] ?? progress,
  );
  if (Math.abs(nearest - progress) <= 0.04) dispatchMovementProgress(nearest);
}

for (const slider of [movementProgress]) {
  slider.addEventListener("pointerdown", stopAnimationForScrubbing);
  slider.addEventListener("input", () => updateMovementProgressFromInput(slider));
  slider.addEventListener("change", () => snapMovementProgressFromInput(slider));
}
window.addEventListener("symmetry-animation-progress-update", event => {
  setMovementProgress(event.detail?.progress || 0);
});
window.addEventListener("symmetry-animation-breakpoints", event => {
  setMovementBreakpoints(event.detail?.breakpoints);
});
window.addEventListener("symmetry-animation-breakpoint-pause", event => {
  setMovementProgress(event.detail?.progress || 0);
  postState({playing: false});
});
const animationSpeed = document.getElementById("animation-speed");
let speedUpdateTimer = null;
animationSpeed.addEventListener("input", () => {
  const speed = Number(animationSpeed.value);
  const output = document.getElementById("animation-speed-value");
  output.value = `${speed.toFixed(2)}×`;
  output.textContent = output.value;
  clearTimeout(speedUpdateTimer);
  speedUpdateTimer = setTimeout(() => postState({speed}), 100);
});
animationSpeed.addEventListener("change", () => {
  clearTimeout(speedUpdateTimer);
  postState({speed: Number(animationSpeed.value)});
});
document.getElementById("pause-at-breakpoints").addEventListener("change", event => {
  const enabled = event.target.checked;
  state.pause_at_breakpoints = enabled;
  syncPlayToggleButton();
  pausePreferenceUpdate = postState({pause_at_breakpoints: enabled}).catch(error => {
    state.pause_at_breakpoints = !enabled;
    syncSpeedButtons();
    showLoadError("Animation setting failed", error);
  });
});
document.getElementById("show-trajectories").addEventListener("change", event => {
  postState({show_trajectories: event.target.checked});
});
document.getElementById("fullscreen-operation-select").addEventListener("change", event => {
  const operationIndex = Number(event.currentTarget.value);
  if (!Number.isInteger(operationIndex)) return;
  activeMode = "standard";
  customUnmappedAtoms = new Set();
  invalidateCustomSequenceWork();
  syncActiveModeControls();
  postState({
    active_mode: "standard",
    operation_index: operationIndex,
    playing: false,
    reset: true,
    clear_custom_check: true,
  });
});
for (const button of document.querySelectorAll(".boundary-button")) {
  button.addEventListener("click", () => postState({
    animation_boundary_mode: button.dataset.boundaryMode,
    playing: false,
    reset: true,
  }));
}
document.getElementById("display-range-decrease").addEventListener("click", () => {
  changeDisplayRange(-1).catch(error => showLoadError("Range update failed", error));
});
document.getElementById("display-range-increase").addEventListener("click", () => {
  changeDisplayRange(1).catch(error => showLoadError("Range update failed", error));
});
document.getElementById("include-boundary-images").addEventListener("change", event => postState({
  include_boundary_images: event.currentTarget.checked,
  playing: false,
  reset: true,
}));
for (const button of document.querySelectorAll(".cell-origin-button")) {
  button.addEventListener("click", () => postState({
    cell_origin_mode: button.dataset.cellOriginMode,
    playing: false,
    reset: true,
  }));
}
for (const button of document.querySelectorAll(".cell-setting-button")) {
  button.addEventListener("click", () => {
    const mode = button.dataset.cellSettingMode;
    if (mode === (state.cell_setting_mode || "native")) return;
    applyCellSetting(mode).catch(error => {
      showLoadError("Cell setting failed", error && error.message ? error.message : error, {kind: "crystal"});
    });
  });
}
for (const button of document.querySelectorAll(".projection-button")) {
  button.addEventListener("click", () => postState({
    projection_mode: button.dataset.projectionMode,
    playing: false,
  }));
}
for (const button of document.querySelectorAll(".background-button")) {
  button.addEventListener("click", () => postState({
    background_mode: button.dataset.backgroundMode,
    playing: false,
  }));
}
for (const button of document.querySelectorAll(".legend-button")) {
  button.addEventListener("click", () => postState({
    legend_visible: button.dataset.legendVisible === "true",
    playing: false,
  }));
}
document.getElementById("improper-mode").addEventListener("change", event => {
  postState({
    improper_mode: event.target.value,
    playing: false,
    reset: true,
  });
});
document.getElementById("view-direction").addEventListener("click", () => {
  postState({view_request_id: Date.now()});
});
document.getElementById("reset-view").addEventListener("click", () => {
  postState({reset_view_request_id: Date.now()});
});
document.getElementById("apply-view-center").addEventListener("click", applyViewCenter);
document.getElementById("view-direction-index").addEventListener("click", applyViewDirectionIndex);
document.getElementById("view-plane-index").addEventListener("click", applyViewPlaneIndex);
document.getElementById("save-gif").addEventListener("click", async () => {
  if (activeMode === "custom") {
    if (!await sendCurrentCustomAnimation(false)) return;
  } else {
    await postState({playing: false});
  }
  window.dispatchEvent(new CustomEvent("symmetry-save-gif"));
});
document.getElementById("save-png").addEventListener("click", () => {
  window.dispatchEvent(new CustomEvent("symmetry-save-png"));
});
function cameraAngle() {
  const value = Number(document.getElementById("camera-angle").value);
  if (!Number.isFinite(value)) return 90;
  return Math.max(0, Math.min(value, 180));
}
function rotateCamera(direction) {
  postState({
    camera_request_id: Date.now(),
    camera_direction: direction,
    camera_angle: cameraAngle(),
  });
}
document.getElementById("camera-left").addEventListener("click", () => rotateCamera("left"));
document.getElementById("camera-right").addEventListener("click", () => rotateCamera("right"));
document.getElementById("camera-up").addEventListener("click", () => rotateCamera("up"));
document.getElementById("camera-down").addEventListener("click", () => rotateCamera("down"));
document.getElementById("camera-roll-left").addEventListener("click", () => rotateCamera("roll-left"));
document.getElementById("camera-roll-right").addEventListener("click", () => rotateCamera("roll-right"));
document.getElementById("displayed-atoms").addEventListener("click", () => {
  postState({selected_atoms: atoms.map(atom => atom.index), scope: "displayed", playing: false, reset: true});
});
document.getElementById("unit-cell-atoms").addEventListener("click", () => {
  postState({selected_atoms: atoms.map(atom => atom.index), scope: "unit_cell", playing: false, reset: true});
});
document.getElementById("clear-atoms").addEventListener("click", () => {
  postState({selected_atoms: [], scope: "displayed", playing: false, reset: true});
});
document.getElementById("reset-atom-appearance").addEventListener("click", resetAtomAppearance);

// --- Custom Operation Check ---

document.getElementById("cop-type").addEventListener("change", () => {
  const type = document.getElementById("cop-type").value;
  for (const el of document.querySelectorAll(".cop-fields")) el.hidden = true;
  const target = document.getElementById("cop-" + type);
  if (target) target.hidden = false;
});

function copNum(id) {
  const input = document.getElementById(id);
  const raw = input.value.trim();
  const value = Number(raw);
  if (!raw || !Number.isFinite(value) || !input.validity.valid) {
    throw new Error(`Enter a valid value for ${input.labels?.[0]?.textContent?.trim() || id}.`);
  }
  return value;
}

function buildCopPayload() {
  const type = document.getElementById("cop-type").value;
  const tolerance = copNum("cop-tol", 0.1);
  let params = {};
  if (type === "rotation") {
    params = {
      axis: [copNum("cop-rot-u"), copNum("cop-rot-v"), copNum("cop-rot-w", 1)],
      angle: copNum("cop-rot-angle", 90),
      point: [copNum("cop-rot-px"), copNum("cop-rot-py"), copNum("cop-rot-pz")],
    };
  } else if (type === "mirror") {
    params = {
      normal: [copNum("cop-mir-h"), copNum("cop-mir-k"), copNum("cop-mir-l", 1)],
      point: [copNum("cop-mir-px"), copNum("cop-mir-py"), copNum("cop-mir-pz")],
    };
  } else if (type === "inversion") {
    params = {
      center: [copNum("cop-inv-cx"), copNum("cop-inv-cy"), copNum("cop-inv-cz")],
    };
  } else if (type === "screw") {
    params = {
      axis: [copNum("cop-scr-u"), copNum("cop-scr-v"), copNum("cop-scr-w", 1)],
      angle: copNum("cop-scr-angle", 180),
      screw: [copNum("cop-scr-tx"), copNum("cop-scr-ty"), copNum("cop-scr-tz", 0.5)],
      point: [copNum("cop-scr-px"), copNum("cop-scr-py"), copNum("cop-scr-pz")],
    };
  } else if (type === "glide") {
    params = {
      normal: [copNum("cop-gli-h"), copNum("cop-gli-k"), copNum("cop-gli-l", 1)],
      point: [copNum("cop-gli-px"), copNum("cop-gli-py"), copNum("cop-gli-pz")],
      glide: [copNum("cop-gli-tx", 0.5), copNum("cop-gli-ty"), copNum("cop-gli-tz")],
    };
  } else if (type === "rotoinversion") {
    params = {
      axis: [copNum("cop-riv-u"), copNum("cop-riv-v"), copNum("cop-riv-w", 1)],
      angle: copNum("cop-riv-angle", 90),
      center: [copNum("cop-riv-cx"), copNum("cop-riv-cy"), copNum("cop-riv-cz")],
    };
  } else if (type === "translation") {
    params = {
      vector: [copNum("cop-tra-x", 0.5), copNum("cop-tra-y", 0.5), copNum("cop-tra-z")],
    };
  } else if (type === "matrix") {
    params = {
      W: [
        copNum("cop-mat-w00",1), copNum("cop-mat-w01"), copNum("cop-mat-w02"),
        copNum("cop-mat-w10"), copNum("cop-mat-w11",1), copNum("cop-mat-w12"),
        copNum("cop-mat-w20"), copNum("cop-mat-w21"), copNum("cop-mat-w22",1),
      ],
      t: [copNum("cop-mat-tx"), copNum("cop-mat-ty"), copNum("cop-mat-tz")],
    };
  }
  return {type, params, tolerance, request_id: Date.now()};
}

function setCopMatrixInputs(W, t) {
  const flatW = (W || []).flat();
  const ids = [
    "cop-mat-w00", "cop-mat-w01", "cop-mat-w02",
    "cop-mat-w10", "cop-mat-w11", "cop-mat-w12",
    "cop-mat-w20", "cop-mat-w21", "cop-mat-w22",
  ];
  ids.forEach((id, index) => {
    const input = document.getElementById(id);
    if (input && Number.isFinite(Number(flatW[index]))) input.value = formatMatrixNumber(flatW[index]);
  });
  ["cop-mat-tx", "cop-mat-ty", "cop-mat-tz"].forEach((id, index) => {
    const input = document.getElementById(id);
    if (input && Number.isFinite(Number((t || [])[index]))) input.value = formatMatrixNumber(t[index]);
  });
  document.getElementById("cop-type").value = "matrix";
  document.getElementById("cop-type").dispatchEvent(new Event("change"));
}

function formatMatrixNumber(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "0";
  const rounded = Math.round(numeric);
  if (Math.abs(numeric - rounded) < 1e-10) return String(rounded);
  return numeric.toFixed(10).replace(/0+$/, "").replace(/\\.$/, "");
}

let copMatrix = null;        // {W_frac, t_frac, op_type, op_params}
let copSelectedAtoms = new Set();
let copAllUnmapped = [];

function displayCopResult(result, opType, opParams) {
  const div = document.getElementById("cop-result");
  copMatrix = null;
  copSelectedAtoms = new Set();
  copAllUnmapped = [];
  customUnmappedAtoms = new Set();
  if (!result) { div.hidden = true; return; }
  if (result.error) {
    div.className = "cop-result fail";
    div.innerHTML = `<strong>Error:</strong> ${result.error}`;
    div.hidden = false;
    return false;
  }
  if (Number.isInteger(result.matching_operation_index) && result.W_frac && result.t_frac) {
    copMatrix = {W_frac: result.W_frac, t_frac: result.t_frac, op_type: opType || "matrix", op_params: opParams || {}, result};
    if (opType === "sequence") setCopMatrixInputs(result.W_frac, result.t_frac);
    div.className = "cop-result";
    div.textContent = `既存の操作 op ${result.matching_operation_index} と同じです。`;
    div.hidden = false;
    renderAtoms();
    renderOperationDetails();
    return true;
  }
  const ok = result.is_symmetry;
  div.className = "cop-result " + (ok ? "ok" : "fail");
  let html = ok
    ? `<strong>✓ All ${result.total} atoms map correctly — this IS a symmetry operation</strong>`
    : result.matrix_issue
      ? `<strong>✗ The matrix is NOT a symmetry operation</strong>`
      : `<strong>✗ ${result.unmapped_count} / ${result.total} atoms do not map — NOT a symmetry operation</strong>`;
  if (result.matrix_issue) html += `<div class="matrix-issue">${result.matrix_issue} It can still be animated as an affine transform.</div>`;

  const coordinateText = values => Array.isArray(values)
    ? `(${values.map(value => Number(value).toFixed(4).replace(/\.?0+$/, "")).join(", ")})`
    : "—";
  if (Array.isArray(result.mapped) && result.mapped.length) {
    html += `<details class="mapping-result-details"><summary>Matched atoms (${result.mapped.length})</summary>`;
    html += `<div class="unmapped-list">`;
    for (const item of result.mapped) {
      html += `<div>atom ${item.source}: <b>${item.element}</b> ${coordinateText(item.source_frac)} → ${coordinateText(item.transformed_frac)} → atom ${item.target} ${coordinateText(item.target_frac)}</div>`;
    }
    html += `</div></details>`;
  }

  if (!ok && result.unmapped && result.unmapped.length > 0) {
    copAllUnmapped = result.unmapped.map(u => u.source);
    customUnmappedAtoms = new Set(copAllUnmapped);
    html += `<details class="mapping-result-details" open><summary>Unmatched atoms (${result.unmapped.length})</summary>`;
    html += `<div class="unmapped-list" id="cop-unmapped-list">`;
    for (const u of result.unmapped) {
      html += `<div>atom ${u.source}: <b>${u.element}</b> ${coordinateText(u.source_frac)} → ${coordinateText(u.transformed_frac)} · nearest atom ${u.nearest_atom ?? "—"} ${coordinateText(u.nearest_frac)} · ${u.distance.toFixed(3)} Å</div>`;
    }
    html += `</div></details>`;
  }
  if (result.W_frac && result.t_frac) {
    copMatrix = {W_frac: result.W_frac, t_frac: result.t_frac, op_type: opType || "matrix", op_params: opParams || {}, result};
    if (opType === "sequence") setCopMatrixInputs(result.W_frac, result.t_frac);
  }
  div.innerHTML = html;
  div.hidden = false;
  renderAtoms();
  renderOperationDetails();
}

function currentAnimationAtoms() {
  const all = atoms.map(atom => atom.index);
  if (state.scope === "representative") return all.slice(0, 1);
  if (state.scope === "unit_cell") return all;
  if (state.scope === "displayed") return all;
  if (String(state.scope).startsWith("selected")) return state.selected_atoms && state.selected_atoms.length ? state.selected_atoms : selectedAtomIndices();
  return all;
}

function currentAnimationUnitCellOnly() {
  return state.scope === "unit_cell" || state.scope === "representative" || state.scope === "selected";
}

function sendCurrentCustomAnimation(startPlaying = true) {
  const result = document.getElementById("cop-result");
  if (!copMatrix) {
    result.className = "cop-result fail";
    result.textContent = "Check the operation sequence first.";
    result.hidden = false;
    return false;
  }
  const indices = currentAnimationAtoms();
  if (!indices.length) {
    result.className = "cop-result fail";
    result.textContent = "No atoms selected.";
    result.hidden = false;
    return false;
  }
  return sendCopAnimate(indices, currentAnimationUnitCellOnly(), startPlaying).then(() => true);
}

async function sendCopAnimate(atomIndices, unitCellOnly = false, startPlaying = true) {
  if (!copMatrix || atomIndices.length === 0) {
    return;
  }
  const body = {
    custom_op_animate: {
      atom_indices: atomIndices,
      W_frac: copMatrix.W_frac,
      t_frac: copMatrix.t_frac,
      op_type: copMatrix.op_type,
      op_params: copMatrix.op_params,
      sequence_items: copMatrix.result && copMatrix.result.sequence_items ? copMatrix.result.sequence_items : null,
      unit_cell_only: Boolean(unitCellOnly),
      animate_id: Date.now(),
    },
    playing: Boolean(startPlaying),
  };
  await postState(body);
}

function customSequencePayloadItems() {
  return customOperationSequence.map(item => {
    if (item.type === "operation") {
      return {type: "operation", index: item.index};
    }
    return {
      type: "custom",
      label: item.label,
      W_frac: item.W_frac,
      t_frac: item.t_frac,
      op_type: item.op_type,
      op_params: item.op_params,
    };
  });
}

async function sendCustomSequenceCompose() {
  const resultDiv = document.getElementById("cop-result");
  if (!customOperationSequence.length) {
    resultDiv.className = "cop-result fail";
    resultDiv.innerHTML = "<strong>Error:</strong> Add one or more operations first.";
    resultDiv.hidden = false;
    return;
  }
  let payload;
  try {
    payload = {
      sequence_items: customSequencePayloadItems(),
      tolerance: copNum("cop-tol"),
      store_custom_result: true,
      request_id: Date.now(),
    };
  } catch (err) {
    resultDiv.className = "cop-result fail";
    resultDiv.innerHTML = `<strong>Error:</strong> ${err.message || err}`;
    resultDiv.hidden = false;
    return;
  }
  resultDiv.className = "cop-result";
  resultDiv.innerHTML = "Composing…";
  resultDiv.hidden = false;
  let sequenceRevision;
  let structureKey;
  let requestGeneration;
  try {
    if (activeMode !== "custom") await setActiveMode("custom");
    sequenceRevision = customSequenceRevision;
    structureKey = `${state.json_path || ""}|${state.reload_request_id || ""}`;
    requestGeneration = ++customComposeRequestGeneration;
    copMatrix = null;
    const result = await api("/api/compose_operations", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    if (!customSequenceContextIsCurrent(sequenceRevision, structureKey)
      || requestGeneration !== customComposeRequestGeneration) return;
    displayCopResult(result, "sequence", {sequence_items: payload.sequence_items});
    if (!result.error) {
      const updatedState = await api("/api/state", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({custom_op_check_id: payload.request_id}),
      });
      if (customSequenceContextIsCurrent(sequenceRevision, structureKey)
        && requestGeneration === customComposeRequestGeneration) {
        state = updatedState;
      }
    }
  } catch (err) {
    if (sequenceRevision !== undefined
      && (!customSequenceContextIsCurrent(sequenceRevision, structureKey)
        || requestGeneration !== customComposeRequestGeneration)) return;
    resultDiv.className = "cop-result fail";
    resultDiv.innerHTML = `<strong>Error:</strong> ${err}`;
  }
}

document.getElementById("btn-add-existing-op").addEventListener("click", () => {
  const select = document.getElementById("cop-operation-select");
  const index = Number(select.value);
  if (Number.isInteger(index)) {
    customOperationSequence.push({type: "operation", index});
    invalidateCustomSequenceWork();
    renderCustomSequenceControls();
  }
});
document.getElementById("btn-add-custom-operation").addEventListener("click", async () => {
  await addCurrentCustomOperationToSequence();
});

async function addCurrentCustomOperationToSequence() {
  const resultDiv = document.getElementById("cop-result");
  const sequenceRevision = customSequenceRevision;
  const structureKey = `${state.json_path || ""}|${state.reload_request_id || ""}`;
  try {
    const payload = buildCopPayload();
    const result = await api("/api/check_operation", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({...payload, build_only: true}),
    });
    if (result.error || !result.W_frac || !result.t_frac) throw new Error(result.error || "Could not build operation");
    if (!customSequenceContextIsCurrent(sequenceRevision, structureKey)) return;
    customOperationSequence.push({
      type: "custom",
      label: `custom ${payload.type}`,
      W_frac: result.W_frac,
      t_frac: result.t_frac,
      op_type: payload.type,
      op_params: payload.params,
    });
    invalidateCustomSequenceWork();
    renderCustomSequenceControls();
  } catch (err) {
    if (!customSequenceContextIsCurrent(sequenceRevision, structureKey)) return;
    resultDiv.className = "cop-result fail";
    resultDiv.innerHTML = `<strong>Error:</strong> ${err}`;
    resultDiv.hidden = false;
  }
}
document.getElementById("btn-compose-custom-sequence").addEventListener("click", sendCustomSequenceCompose);
document.getElementById("btn-clear-custom-sequence").addEventListener("click", () => {
  customOperationSequence = [];
  invalidateCustomSequenceWork();
  renderCustomSequenceControls();
});
async function boot() {
  const st = document.getElementById("status");
  syncExperienceModeControls();
  st.textContent = "Connecting…";
  const [info, atomInfo, stateInfo, examplesInfo] = await Promise.all([
    api("/api/operations"),
    api("/api/atoms"),
    api("/api/state"),
    api("/api/examples"),
  ]);
  operations = info.operations;
  summariesReady = Boolean(info.summaries_ready);
  st.textContent = `Loaded ${operations.length} operations`;
  atoms = atomInfo.atoms;
  st.textContent = `Loaded ${operations.length} operations, ${atoms.length} atoms`;
  state = stateInfo;
  exampleCatalog = examplesInfo || {crystal: [], molecule: []};
  if (state.source_kind === "crystal" || state.source_kind === "molecule") {
    selectedStructureKind = state.source_kind;
  }
  sourceKind = state.source_kind || "crystal";
  setDefaultOperationSortForSourceKind();
  await refreshAtomMotion();
  syncSourceKindControls();
  renderDirectionFilter();
  renderAtomElementFilter();
  renderElementColorControls();
  syncSpeedButtons();
  syncBoundaryButtons();
  syncDisplayButtons();
  syncCellOriginButtons();
  syncCellSettingButtons();
  syncProjectionButtons();
  syncBackgroundButtons();
  syncLegendButtons();
  syncImproperModeControl();
  syncAtomModeButtons();
  syncPlayToggleButton();
  renderOperations();
  renderAtoms();
  renderSelectedAtomSummary();
  renderStructureInfo();
  renderStatus();
  renderOperationDetails();
  analysisBootComplete = true;
  setInterval(() => {
    refreshState().catch(error => {
      document.getElementById("status").textContent = `Refresh error: ${error}`;
    });
  }, 1000);
}
boot().catch(error => {
  document.getElementById("status").textContent = `Boot error: ${error}`;
});

async function refreshAnalysisFromServer() {
  if (!analysisBootComplete || analysisRefreshInProgress) return;
  analysisRefreshInProgress = true;
  try {
    const [info, atomInfo, stateInfo, examplesInfo] = await Promise.all([
      api("/api/operations"),
      api("/api/atoms"),
      api("/api/state"),
      api("/api/examples"),
    ]);
    operations = info.operations || [];
    summariesReady = Boolean(info.summaries_ready);
    atoms = atomInfo.atoms || [];
    state = stateInfo || {};
    exampleCatalog = examplesInfo || {crystal: [], molecule: []};
    if (state.source_kind === "crystal" || state.source_kind === "molecule") {
      selectedStructureKind = state.source_kind;
    }
    sourceKind = state.source_kind || "crystal";
    // Back to hub resets the server to its empty session.  This path bypasses
    // applyLoadedStructure(), so discard client-only Custom state here too;
    // otherwise re-entering Analysis leaves an old sequence/result visible
    // even though there is no structure to which it can belong.
    if (state.structure_loaded === false || sourceKind === "empty") {
      customUnmappedAtoms = new Set();
      customOperationSequence = [];
      invalidateCustomSequenceWork();
      activeMode = "standard";
      syncActiveModeControls();
    }
    setDefaultOperationSortForSourceKind();
    await refreshAtomMotion();
    syncSourceKindControls();
    renderExampleOptions();
    renderDirectionFilter();
    renderAtomElementFilter();
    renderElementColorControls();
    syncOperationSelection();
    syncSpeedButtons();
    syncBoundaryButtons();
    syncDisplayButtons();
    syncCellOriginButtons();
    syncCellSettingButtons();
    syncProjectionButtons();
    syncBackgroundButtons();
    syncLegendButtons();
    syncImproperModeControl();
    syncAtomModeButtons();
    syncPlayToggleButton();
      renderOperations();
    renderAtoms();
    renderSelectedAtomSummary();
    renderStructureInfo();
    renderStatus();
    renderOperationDetails();
    if (window.symmetryThreeView) {
      await window.symmetryThreeView.refresh();
      await window.symmetryThreeView.syncState(state, {structure_reload: true});
      window.symmetryThreeView.lastStateSignature = `${state.json_path || ""}|${state.reload_request_id || 0}`;
    }
  } catch (error) {
    document.getElementById("status").textContent = `Analysis refresh error: ${error}`;
  } finally {
    analysisRefreshInProgress = false;
  }
}

// --- Top-level Analysis / Puzzle mode (docs/PUZZLE_SPEC.md §1) ---
// Selection screen is the entry point; you return here with Back before
// switching, so puzzle and analysis never bleed into each other.
function readInitialAppMode() {
  try {
    const mode = JSON.parse(document.getElementById("app-mode-config").textContent);
    return mode === "analysis" || mode === "puzzle" ? mode : "select";
  } catch (error) {
    return "select";
  }
}
let appMode = readInitialAppMode();
// Whether the current analysis-mode visit was entered via puzzle.js's
// openCurrentStructureInAnalysis() shortcut (point-group quiz's "解析モードで
// 開く") rather than the hub's 解析モード button. #analysis-back used to have a
// sibling #analysis-back-to-puzzle button for this case, but that button had
// its own independent hidden toggle while #analysis-back has none (it is
// shown/hidden only by whether .topbar itself is covered by a mode overlay),
// so a puzzle-shortcut visit showed both buttons side by side at once. One
// button that relabels itself avoids that by construction.
let analysisReturnsToPuzzle = false;
function updateAnalysisBackButton() {
  document.getElementById("analysis-back").textContent = analysisReturnsToPuzzle ? "← クイズに戻る" : "← 戻る";
}
function applyAppMode() {
  document.getElementById("mode-select").hidden = appMode !== "select";
  document.getElementById("puzzle-mode").hidden = appMode !== "puzzle";
  document.body.classList.toggle("in-puzzle", appMode === "puzzle");
  document.body.classList.toggle("analysis-theme", appMode === "analysis");
  if (appMode === "analysis") {
    // The 3D canvas may have initialized while covered; nudge a resize.
    window.dispatchEvent(new Event("resize"));
    refreshAnalysisFromServer();
  }
  if (appMode === "puzzle") {
    // puzzle.js lazily builds its own 3D view the first time this fires.
    window.dispatchEvent(new Event("symmetry-enter-puzzle"));
  } else {
    // Tell the puzzle's view to stop drawing; without this both renderers keep
    // an animation frame loop running for as long as the page is open.
    window.dispatchEvent(new Event("symmetry-exit-puzzle"));
  }
  window.symmetryThreeView?.setActive(appMode !== "puzzle");
}
function setAppMode(mode, { fromPuzzle = false } = {}) {
  appMode = mode;
  // Only openCurrentStructureInAnalysis() (puzzle.js) passes fromPuzzle: true,
  // right when calling setAppMode("analysis") itself; every other way of
  // changing mode resets it so it does not linger into an unrelated analysis
  // visit.
  analysisReturnsToPuzzle = mode === "analysis" && fromPuzzle;
  updateAnalysisBackButton();
  applyAppMode();
}
document.getElementById("enter-analysis").addEventListener("click", () => setAppMode("analysis"));
document.getElementById("enter-puzzle").addEventListener("click", () => setAppMode("puzzle"));
document.getElementById("analysis-back").addEventListener("click", async () => {
  if (analysisReturnsToPuzzle) {
    resumePuzzleMode();
    return;
  }
  try {
    await api("/api/reset_analysis", {method: "POST", body: "{}"});
  } catch (error) {
    document.getElementById("status").textContent = `Reset error: ${error}`;
  }
  setAppMode("select");
});
// #puzzle-back's own click handler now lives in puzzle.js (goBack()): within
// puzzle mode it means "one step back", which only sometimes means the hub, so
// it needs puzzle.js's own screen state to resolve — see
// docs/sessions/SESSION_REPORT_2026-09-09.md.

// Puzzle mode's point-group quiz can jump straight into analysis mode on the
// same structure (openCurrentStructureInAnalysis() in puzzle.js) without going
// through 戻る/select first, so unlike every other path here, the puzzle side's
// screen (and its round state) is still intact underneath and just needs
// re-showing — not enterPuzzle()'s usual reset to the quiz list. A dedicated
// event (rather than reusing symmetry-enter-puzzle) keeps that distinction
// explicit for puzzle.js's own listener.
function resumePuzzleMode() {
  appMode = "puzzle";
  document.getElementById("mode-select").hidden = true;
  document.getElementById("puzzle-mode").hidden = false;
  document.body.classList.add("in-puzzle");
  document.body.classList.remove("analysis-theme");
  analysisReturnsToPuzzle = false;
  updateAnalysisBackButton();
  window.dispatchEvent(new Event("symmetry-resume-puzzle"));
  window.symmetryThreeView?.setActive(false);
}
applyAppMode();
