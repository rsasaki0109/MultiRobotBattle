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

function drawElevation(ctx, X, Y, sc, cx, cy, hw, hh) {
  ctx.fillStyle = "rgba(18,24,32,0.55)";
  ctx.strokeStyle = "#2a3848";
  ctx.lineWidth = Math.max(0.7, sc * 0.7);
  ctx.fillRect(X(cx - hw), Y(cy + hh), 2 * hw * sc, -2 * hh * sc);
  ctx.strokeRect(X(cx - hw), Y(cy + hh), 2 * hw * sc, -2 * hh * sc);
  const inset = 0.35;
  ctx.strokeStyle = "rgba(61,80,104,0.45)";
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(
    X(cx - hw + inset),
    Y(cy + hh - inset),
    2 * (hw - inset) * sc,
    -2 * (hh - inset) * sc,
  );
  ctx.setLineDash([]);
  const cap = Math.min(hw, hh) * 0.55;
  ctx.fillStyle = "rgba(245,204,77,0.18)";
  ctx.fillRect(
    X(cx - cap * 0.5),
    Y(cy + hh - cap * 0.35),
    cap * sc,
    -cap * 0.18 * sc,
  );
}

function drawWall(ctx, X, Y, sc, cx, cy, hw, hh) {
  ctx.fillStyle = "#1a222c";
  ctx.strokeStyle = "#3a4a62";
  ctx.lineWidth = Math.max(1.2, sc * 1.2);
  ctx.fillRect(X(cx - hw), Y(cy + hh), 2 * hw * sc, -2 * hh * sc);
  ctx.strokeRect(X(cx - hw), Y(cy + hh), 2 * hw * sc, -2 * hh * sc);
  const stripeH = Math.min(hh * 0.28, 0.55) * sc;
  ctx.fillStyle = "rgba(245,204,77,0.32)";
  ctx.fillRect(
    X(cx - hw * 0.92),
    Y(cy + hh),
    hw * 1.84 * sc,
    -stripeH,
  );
  ctx.fillStyle = "rgba(14,18,24,0.5)";
  ctx.fillRect(
    X(cx - hw * 0.75),
    Y(cy + hh * 0.45),
    hw * 1.5 * sc,
    -hh * 0.12 * sc,
  );
}

function drawObstacle(ctx, X, Y, sc, ox, oy, r) {
  ctx.fillStyle = "#222a32";
  ctx.strokeStyle = "#33405a";
  ctx.lineWidth = Math.max(1, sc * 1.1);
  ctx.beginPath();
  ctx.arc(X(ox), Y(oy), r * sc, 0, 7);
  ctx.fill();
  ctx.stroke();
  if (r >= 2.2) {
    ctx.fillStyle = "#161b24";
    ctx.strokeStyle = "#2a3448";
    ctx.lineWidth = Math.max(0.6, sc * 0.6);
    ctx.beginPath();
    ctx.arc(X(ox), Y(oy), r * 0.62 * sc, 0, 7);
    ctx.fill();
    ctx.stroke();
    const capW = r * 0.95 * sc;
    const capH = r * 0.22 * sc;
    ctx.fillStyle = "rgba(245,204,77,0.28)";
    ctx.fillRect(X(ox) - capW * 0.5, Y(oy + r * 0.42 * sc) - capH, capW, capH);
  } else if (r >= 1.4) {
    ctx.strokeStyle = "#3d4a62";
    ctx.lineWidth = Math.max(0.5, sc * 0.5);
    ctx.beginPath();
    ctx.arc(X(ox), Y(oy), r * 0.45 * sc, 0, 7);
    ctx.stroke();
  }
}

function drawProjectile(ctx, X, Y, sc, px, py, team) {
  const [r, g, b] = TEAM_RGB[team] || [0.95, 0.95, 0.95];
  ctx.fillStyle = rgbaStr([r, g, b, 0.95]);
  ctx.beginPath();
  ctx.arc(X(px), Y(py), Math.max(1.5, sc * 0.35), 0, 7);
  ctx.fill();
  ctx.fillStyle = rgbaStr([1, 1, 1, 0.55]);
  ctx.beginPath();
  ctx.arc(X(px), Y(py), Math.max(0.8, sc * 0.15), 0, 7);
  ctx.fill();
}

function drawPayload(ctx, X, Y, sc, px, py, team) {
  const [r, g, b] = TEAM_RGB[team] || [0.96, 0.82, 0.3];
  const w = Math.max(10, sc * 2.2);
  const h = Math.max(7, sc * 1.5);
  const cx = X(px);
  const cy = Y(py);
  ctx.fillStyle = rgbaStr([r * 0.55 + 0.2, g * 0.55 + 0.2, b * 0.55 + 0.2, 0.92]);
  ctx.strokeStyle = "#f5cc4d";
  ctx.lineWidth = Math.max(1.2, sc * 0.35);
  ctx.beginPath();
  ctx.roundRect(cx - w * 0.5, cy - h * 0.5, w, h, Math.max(2, sc * 0.25));
  ctx.fill();
  ctx.stroke();
  ctx.strokeStyle = "#0f1320";
  ctx.lineWidth = Math.max(1.5, sc * 0.4);
  ctx.beginPath();
  ctx.moveTo(cx - w * 0.25, cy);
  ctx.lineTo(cx + w * 0.25, cy);
  ctx.stroke();
}

window.BattleRobotArt = {
  inferHeading,
  drawRobotChassis,
  drawShot,
  drawProjectile,
  drawPayload,
  drawElevation,
  drawWall,
  drawObstacle,
};
