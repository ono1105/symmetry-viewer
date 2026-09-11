const EPSILON = 1e-12;

function add(a, b) {
  return a.map((value, index) => value + b[index]);
}

function subtract(a, b) {
  return a.map((value, index) => value - b[index]);
}

function scale(vector, factor) {
  return vector.map(value => value * factor);
}

function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize(vector) {
  const norm = Math.sqrt(dot(vector, vector));
  return norm < EPSILON ? [...vector] : scale(vector, 1 / norm);
}

function interpolate(start, target, s) {
  return start.map((value, index) => (1 - s) * value + s * target[index]);
}

function rotateAboutAxis(point, axisPoint, axisDirection, angleRad) {
  const direction = normalize(axisDirection);
  const relative = subtract(point, axisPoint);
  const cosAngle = Math.cos(angleRad);
  const sinAngle = Math.sin(angleRad);
  const rotated = add(
    add(scale(relative, cosAngle), scale(cross(direction, relative), sinAngle)),
    scale(direction, dot(direction, relative) * (1 - cosAngle)),
  );
  return add(axisPoint, rotated);
}

function reflectPoint(point, planePoint, planeNormal) {
  const normal = normalize(planeNormal);
  return subtract(point, scale(normal, 2 * dot(subtract(point, planePoint), normal)));
}

function multiplyMatrixVector(matrix, vector) {
  return matrix.map(row => dot(row, vector));
}

function multiplyRowVectorMatrix(vector, matrix) {
  return matrix[0].map((_, column) =>
    vector.reduce((sum, value, row) => sum + value * matrix[row][column], 0));
}

export function applyBoundaryContext(position, context) {
  if (!context || context.mode !== "wrap") return [...position];
  let cell = multiplyRowVectorMatrix(position, context.cart_to_cell);
  const epsilon = Number(context.boundary_epsilon ?? 1e-9);
  if (context.cell_origin_mode === "corner") {
    cell = cell.map(value => value - Math.floor(value));
    cell = cell.map(value => value >= 1 - epsilon ? value - 1 : value);
  } else {
    cell = cell.map(value => value - Math.floor(value + 0.5));
    cell = cell.map(value => value >= 0.5 - epsilon ? value - 1 : value);
  }
  return multiplyRowVectorMatrix(cell, context.cell_to_cart);
}

export function pathAppliesToDisplayInstance(path, instance) {
  return Boolean(path) && (!path.unit_cell_only || Boolean(instance.isPrimaryImage));
}

function angleRadians(path) {
  if (!Number.isFinite(path.angle_deg)) {
    throw new Error(`Path type ${path.type} requires angle_deg`);
  }
  return path.angle_deg * Math.PI / 180;
}

function rotationArcLength(path, start) {
  const direction = normalize(path.axis_direction);
  const relative = subtract(start, path.axis_point);
  const axial = scale(direction, dot(relative, direction));
  return Math.sqrt(dot(subtract(relative, axial), subtract(relative, axial))) * Math.abs(angleRadians(path));
}

function twoPhaseGeometry(path, start) {
  let midpoint;
  let endpoint;
  let firstLength;
  if (path.type === "glide") {
    midpoint = reflectPoint(start, path.plane_point, path.plane_normal);
    endpoint = add(midpoint, path.translation);
    firstLength = Math.sqrt(dot(subtract(midpoint, start), subtract(midpoint, start)));
  } else {
    midpoint = rotateAboutAxis(start, path.axis_point, path.axis_direction, angleRadians(path));
    firstLength = rotationArcLength(path, start);
    if (path.type === "screw") endpoint = add(midpoint, path.translation);
    else if (path.type === "rotoinversion") endpoint = subtract(scale(path.center, 2), midpoint);
    else endpoint = reflectPoint(midpoint, path.plane_point, path.plane_normal);
  }
  const secondDelta = subtract(endpoint, midpoint);
  return {midpoint, endpoint, firstLength, secondLength: Math.sqrt(dot(secondDelta, secondDelta))};
}

function pathLength(path, startOverride = null) {
  if (path.type === "sequential") {
    if (!path.segments?.length) throw new Error("Sequential path requires at least one segment");
    let segmentStart = startOverride === null ? [...path.segments[0].start] : [...startOverride];
    let length = 0;
    for (const segment of path.segments) {
      length += pathLength(segment, segmentStart);
      segmentStart = evaluatePath(segment, 1, segmentStart);
    }
    return length;
  }
  const start = startOverride === null ? [...path.start] : [...startOverride];
  if (path.type === "rotation") return rotationArcLength(path, start);
  if (["screw", "glide", "rotoinversion", "rotoreflection"].includes(path.type)) {
    const geometry = twoPhaseGeometry(path, start);
    return geometry.firstLength + geometry.secondLength;
  }
  const delta = subtract(evaluatePath(path, 1, start), start);
  return Math.sqrt(dot(delta, delta));
}

function sequentialWeights(path, lengths) {
  const supplied = path.segment_weights;
  if (Array.isArray(supplied) && supplied.length === lengths.length
      && supplied.every(value => Number.isFinite(Number(value)) && Number(value) >= 0)) {
    return supplied.map(Number);
  }
  return lengths;
}

function phaseFraction(firstLength, secondLength) {
  const total = firstLength + secondLength;
  return total > EPSILON ? firstLength / total : 0.5;
}

export function pathBreakpoints(path, startOverride = null) {
  const result = new Set([0, 1]);
  if (path.type === "sequential" && path.segments?.length) {
    let segmentStart = startOverride === null ? [...path.segments[0].start] : [...startOverride];
    const starts = [];
    const lengths = [];
    for (const segment of path.segments) {
      starts.push(segmentStart);
      lengths.push(pathLength(segment, segmentStart));
      segmentStart = evaluatePath(segment, 1, segmentStart);
    }
    const weights = sequentialWeights(path, lengths);
    const total = weights.reduce((sum, length) => sum + length, 0);
    let cumulative = 0;
    for (let index = 0; index < path.segments.length; index += 1) {
      const scale = total > EPSILON ? weights[index] / total : 1 / weights.length;
      for (const local of pathBreakpoints(path.segments[index], starts[index])) {
        result.add(cumulative + scale * local);
      }
      cumulative += scale;
    }
  } else if (["screw", "glide", "rotoinversion", "rotoreflection"].includes(path.type)) {
    const start = startOverride === null ? [...path.start] : [...startOverride];
    const geometry = twoPhaseGeometry(path, start);
    result.add(Math.max(0, Math.min(
      Number(path.phase_fraction ?? phaseFraction(geometry.firstLength, geometry.secondLength)),
      1,
    )));
  } else if (path.type === "mirror_after_hold") {
    result.add(Math.max(0, Math.min(Number(path.hold_fraction) || 0.3, 1)));
  }
  return [...result].sort((a, b) => a - b);
}

export function sequentialSegmentIndex(path, progress, startOverride = null) {
  if (path.type !== "sequential" || !path.segments?.length) return null;
  const s = Math.min(1, Math.max(0, Number(progress)));
  const boundaries = sequentialSegmentBoundaries(path, startOverride);
  return boundaries.findIndex(boundary => s <= boundary + EPSILON);
}

export function sequentialSegmentBoundaries(path, startOverride = null) {
  if (path.type !== "sequential" || !path.segments?.length) return [];
  let segmentStart = startOverride === null ? [...path.segments[0].start] : [...startOverride];
  const lengths = [];
  for (const segment of path.segments) {
    lengths.push(pathLength(segment, segmentStart));
    segmentStart = evaluatePath(segment, 1, segmentStart);
  }
  const weights = sequentialWeights(path, lengths);
  const total = weights.reduce((sum, length) => sum + length, 0);
  if (total <= EPSILON) return weights.map((_, index) => (index + 1) / weights.length);
  let cumulative = 0;
  return weights.map(weight => {
    cumulative += weight / total;
    return cumulative;
  });
}

export function evaluatePath(path, progress, startOverride = null) {
  const s = Math.min(1, Math.max(0, Number(progress)));
  const pathType = path.type;

  if (pathType === "sequential") {
    const segments = path.segments;
    if (!segments.length) {
      throw new Error("Sequential path requires at least one segment");
    }
    let segmentStart = startOverride === null ? [...segments[0].start] : [...startOverride];
    const starts = [];
    const lengths = [];
    for (const segment of segments) {
      starts.push(segmentStart);
      lengths.push(pathLength(segment, segmentStart));
      segmentStart = evaluatePath(segment, 1, segmentStart);
    }
    const weights = sequentialWeights(path, lengths);
    const total = weights.reduce((sum, length) => sum + length, 0);
    let index;
    let localS;
    if (total <= EPSILON) {
      index = Math.min(Math.floor(s * segments.length), segments.length - 1);
      localS = s * segments.length - index;
    } else {
      const distance = s * total;
      let cumulative = 0;
      index = segments.length - 1;
      localS = 1;
      for (let candidate = 0; candidate < segments.length; candidate += 1) {
        const length = weights[candidate];
        if (distance <= cumulative + length || candidate === segments.length - 1) {
          index = candidate;
          localS = length <= EPSILON ? 1 : (distance - cumulative) / length;
          break;
        }
        cumulative += length;
      }
    }
    return evaluatePath(segments[index], localS, starts[index]);
  }

  const start = startOverride === null ? [...path.start] : [...startOverride];
  if (pathType === "affine_linear") {
    const target = startOverride === null
      ? path.target
      : add(multiplyMatrixVector(path.matrix_cart, start), path.translation_cart);
    return interpolate(start, target, s);
  }
  if (pathType === "rotation") {
    return rotateAboutAxis(start, path.axis_point, path.axis_direction, angleRadians(path) * s);
  }
  if (["screw", "glide", "rotoinversion", "rotoreflection"].includes(pathType)) {
    const geometry = twoPhaseGeometry(path, start);
    const split = Math.max(0, Math.min(
      Number(path.phase_fraction ?? phaseFraction(geometry.firstLength, geometry.secondLength)),
      1,
    ));
    if (s <= split) {
      const localS = split > EPSILON ? s / split : 1;
      if (pathType === "glide") return interpolate(start, geometry.midpoint, localS);
      return rotateAboutAxis(start, path.axis_point, path.axis_direction, angleRadians(path) * localS);
    }
    const localS = split < 1 - EPSILON ? (s - split) / (1 - split) : 1;
    return interpolate(geometry.midpoint, geometry.endpoint, localS);
  }
  if (pathType === "mirror" || pathType === "mirror_after_hold") {
    if (pathType === "mirror_after_hold") {
      const holdFraction = Number(path.hold_fraction ?? 0.3);
      if (s <= holdFraction) {
        return [...start];
      }
      const localS = (s - holdFraction) / Math.max(1 - holdFraction, 1e-9);
      return interpolate(start, reflectPoint(start, path.plane_point, path.plane_normal), localS);
    }
    return interpolate(start, reflectPoint(start, path.plane_point, path.plane_normal), s);
  }
  if (pathType === "inversion") {
    return interpolate(start, subtract(scale(path.center, 2), start), s);
  }

  let target = [...path.target];
  if (startOverride !== null) {
    target = add(target, subtract(start, path.start));
  }
  return interpolate(start, target, s);
}
