/** RoboMaster-style robot silhouettes for the battle browser demo. */
"use strict";

const TEAM_RGB = {
  0: [0.96, 0.28, 0.28],
  1: [0.28, 0.58, 0.98],
  2: [0.28, 0.88, 0.48],
  3: [0.98, 0.82, 0.22],
};

const HULL = {
  "": [1.35, 0.95, 0.18],
  scout: [0.95, 0.62, 0.14],
  soldier: [1.35, 0.95, 0.18],
  tank: [2.15, 1.55, 0.24],
  sniper: [1.15, 0.82, 0.16],
};

const BARREL = {
  "": 0.72,
  scout: 0.55,
  soldier: 0.78,
  tank: 1.05,
  sniper: 1.35,
};

function defaultHeading(team) {
  if (team === 0 || team === 2) return 0;
  if (team === 1 || team === 3) return Math.PI;
  return 0;
}

function inferHeading(x, y, px, py, team) {
  const dx = x - px;
  const dy = y - py;
  if (dx * dx + dy * dy > 2e-3) return Math.atan2(dy, dx);
  return defaultHeading(team);
}

function hullSpec(kind) {
  return HULL[kind] || HULL[""];
}

function hullPolygon(x, y, heading, kind) {
  const [L, W] = hullSpec(kind);
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  const hx = L * 0.5;
  const hy = W * 0.5;
  const local = [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]];
  return local.map(([lx, ly]) => [x + c * lx - s * ly, y + s * lx + c * ly]);
}

function hullFaceRgba(team, hp) {
  const [r, g, b] = TEAM_RGB[team] || [0.7, 0.7, 0.7];
  const base = 0.22 + 0.18 * Math.max(0, Math.min(1, hp));
  return [
    Math.min(1, base + r * 0.55),
    Math.min(1, base + g * 0.55),
    Math.min(1, base + b * 0.55),
    0.88,
  ];
}

function stripePolygon(x, y, heading, kind) {
  const [L, W] = hullSpec(kind);
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  const fx = L * 0.42;
  const hw = W * 0.38;
  const local = [
    [fx, -hw],
    [L * 0.48, -hw * 0.55],
    [L * 0.48, hw * 0.55],
    [fx, hw],
  ];
  return local.map(([lx, ly]) => [x + c * lx - s * ly, y + s * lx + c * ly]);
}

function stripeRgba(team, hp) {
  const [r, g, b] = TEAM_RGB[team] || [0.8, 0.8, 0.8];
  const a = 0.55 + 0.45 * Math.max(0, Math.min(1, hp));
  return [r, g, b, a];
}

function turretCenter(x, y, heading, kind) {
  const [L] = hullSpec(kind);
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  const ox = L * 0.08;
  return [x + c * ox, y + s * ox];
}

function barrelSegment(x, y, heading, kind) {
  const [cx, cy] = turretCenter(x, y, heading, kind);
  const bl = BARREL[kind] ?? BARREL[""];
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  return [[cx, cy], [cx + c * bl, cy + s * bl]];
}

function wheelOffsets(x, y, heading, kind) {
  const [L, W] = hullSpec(kind);
  if (kind && kind !== "soldier" && kind !== "tank") return [];
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  const spread = L * 0.28;
  const side = W * 0.38;
  const pts = [];
  for (const sign of [-1, 1]) {
    for (const ly of [-side, side]) {
      const lx = sign * spread;
      pts.push([x + c * lx - s * ly, y + s * lx + c * ly]);
    }
  }
  return pts;
}

function wheelRadius(kind) {
  if (kind === "tank") return 0.16;
  if (!kind || kind === "soldier") return 0.11;
  return 0;
}

function rgbaStr([r, g, b, a]) {
  return `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${a})`;
}

function fillPoly(ctx, X, Y, sc, pts, rgba) {
  if (!pts.length) return;
  ctx.fillStyle = rgbaStr(rgba);
  ctx.beginPath();
  ctx.moveTo(X(pts[0][0]), Y(pts[0][1]));
  for (let i = 1; i < pts.length; i++) ctx.lineTo(X(pts[i][0]), Y(pts[i][1]));
  ctx.closePath();
  ctx.fill();
}

function drawRobotChassis(ctx, X, Y, sc, x, y, team, hp, kind, heading) {
  const k = kind || "soldier";
  const wr = wheelRadius(k);
  if (wr > 0) {
    for (const [wx, wy] of wheelOffsets(x, y, heading, k)) {
      ctx.fillStyle = "rgba(31,33,41,0.85)";
      ctx.beginPath();
      ctx.ellipse(X(wx), Y(wy), wr * sc, wr * 0.55 * sc, 0, 0, 7);
      ctx.fill();
    }
  }
  fillPoly(ctx, X, Y, sc, hullPolygon(x, y, heading, k), hullFaceRgba(team, hp));
  fillPoly(ctx, X, Y, sc, stripePolygon(x, y, heading, k), stripeRgba(team, hp));
  const [[ax, ay], [bx, by]] = barrelSegment(x, y, heading, k);
  ctx.strokeStyle = "#22262e";
  ctx.lineWidth = Math.max(1, sc * 0.9);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(X(ax), Y(ay));
  ctx.lineTo(X(bx), Y(by));
  ctx.stroke();
}

function drawShot(ctx, X, Y, sc, sx0, sy0, x1, y1, team) {
  const [r, g, b] = TEAM_RGB[team] || [0.9, 0.9, 0.9];
  ctx.lineCap = "round";
  ctx.strokeStyle = rgbaStr([r, g, b, 0.35]);
  ctx.lineWidth = Math.max(2, sc * 2.4);
  ctx.beginPath();
  ctx.moveTo(X(sx0), Y(sy0));
  ctx.lineTo(X(x1), Y(y1));
  ctx.stroke();
  ctx.strokeStyle = rgbaStr([
    Math.min(1, r + 0.25),
    Math.min(1, g + 0.25),
    Math.min(1, b + 0.2),
    0.9,
  ]);
  ctx.lineWidth = Math.max(1, sc * 0.9);
  ctx.beginPath();
  ctx.moveTo(X(sx0), Y(sy0));
  ctx.lineTo(X(x1), Y(y1));
  ctx.stroke();
}

window.BattleRobotArt = {
  inferHeading,
  drawRobotChassis,
  drawShot,
};
