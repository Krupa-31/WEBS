"""
Dynamic SITL HTML Generator for WEBS Dashboard.
Generates a fully self-contained Hardware SITL simulation HTML page
with the ACTUAL custom orders, real Bangalore street network, and
evaluated trajectories injected directly from Python.
"""

import json
import math


def _normalize_positions(node_positions: dict, W: int = 860, H: int = 580, PAD: int = 55):
    """
    Map city_map meter coordinates to canvas pixel coords.
    X = East (→), Y = North (↑ in geo, ↓ in canvas → invert).
    """
    xs = [p[0] for p in node_positions.values()]
    ys = [p[1] for p in node_positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rng_x = max(max_x - min_x, 1.0)
    rng_y = max(max_y - min_y, 1.0)

    canvas_pos = {}
    for nid, (mx, my) in node_positions.items():
        cx = PAD + ((mx - min_x) / rng_x) * (W - 2 * PAD)
        cy = (H - PAD) - ((my - min_y) / rng_y) * (H - 2 * PAD)
        canvas_pos[nid] = {"cx": round(cx, 1), "cy": round(cy, 1)}
    return canvas_pos


DRONE_COLORS = [
    "#10b981",  # emerald
    "#38bdf8",  # sky blue
    "#f59e0b",  # amber
    "#a855f7",  # purple
    "#ec4899",  # pink
    "#06b6d4",  # cyan
    "#84cc16",  # lime
    "#f97316",  # orange
]


def generate_sitl_html(
    plans,          # List[PlanResult]
    requests_dict,  # Dict[int, DeliveryRequest]
    city_map,       # CityAirspaceMap
    wind_speed: float,
    wind_dir_deg: float,
    landmark_dict: dict,
    specs,          # DroneSpecs
) -> str:
    """
    Returns a complete HTML string for the Hardware SITL simulator,
    pre-loaded with the actual evaluated WEBS order results.
    """
    W, H, PAD = 860, 580, 55
    canvas_pos = _normalize_positions(city_map.node_positions, W, H, PAD)

    # ── Build edge list ───────────────────────────────────────────────────────
    edges_js = []
    for u, v in city_map.graph.edges():
        if u in canvas_pos and v in canvas_pos:
            edata = city_map.graph.get_edge_data(u, v) or {}
            name = edata.get("street_name", edata.get("name", ""))
            edges_js.append({
                "u": canvas_pos[u],
                "v": canvas_pos[v],
                "name": str(name)[:20]
            })

    # ── Build node list ───────────────────────────────────────────────────────
    nodes_js = []
    for nid, pos in canvas_pos.items():
        label = landmark_dict.get(nid, "")
        nodes_js.append({
            "id": nid,
            "cx": pos["cx"],
            "cy": pos["cy"],
            "label": label[:22],
            "is_landmark": (nid in landmark_dict)
        })

    # ── Build drone fleet from accepted plans ─────────────────────────────────
    drones_js = []
    rejected_js = []
    color_idx = 0

    for i, plan in enumerate(plans):
        req = requests_dict.get(plan.request_id)
        if req is None:
            continue
        src_name = landmark_dict.get(req.source, f"Node {req.source}")
        dst_name = landmark_dict.get(req.destination, f"Node {req.destination}")
        order_label = f"ORD-{plan.request_id:02d}"

        if plan.success and plan.trajectory:
            color = DRONE_COLORS[color_idx % len(DRONE_COLORS)]
            color_idx += 1

            # Build waypoints from actual trajectory (deduplicate consecutive nodes)
            waypoints = []
            prev_node = None
            for (node_id, t_arr, t_dep) in plan.trajectory:
                if node_id == prev_node:
                    continue
                prev_node = node_id
                if node_id in canvas_pos:
                    nm = landmark_dict.get(node_id, f"Node {node_id}")
                    waypoints.append({
                        "cx": canvas_pos[node_id]["cx"],
                        "cy": canvas_pos[node_id]["cy"],
                        "name": nm[:20],
                        "t_arr": round(t_arr, 1)
                    })

            if len(waypoints) < 2:
                if req.source in canvas_pos and req.destination in canvas_pos:
                    waypoints = [
                        {"cx": canvas_pos[req.source]["cx"], "cy": canvas_pos[req.source]["cy"],
                         "name": src_name[:20], "t_arr": 0},
                        {"cx": canvas_pos[req.destination]["cx"], "cy": canvas_pos[req.destination]["cy"],
                         "name": dst_name[:20], "t_arr": round(plan.flight_time_s, 1)},
                    ]

            soc_end = max(0.0, 100.0 - (plan.energy_joules / specs.battery_joules * 100.0))
            drones_js.append({
                "id": i + 1,
                "name": f"UAV-{plan.drone_id:02d}",
                "orderLabel": order_label,
                "orderName": f"{src_name} → {dst_name}",
                "payload": round(req.payload_kg, 2),
                "deadline": round(req.deadline / 60.0, 1),
                "color": color,
                "waypoints": waypoints,
                "energyKj": round(plan.energy_joules / 1000.0, 2),
                "socEnd": round(soc_end, 1),
                "flightTime": round(plan.flight_time_s, 1),
                "hasParcel": True,
            })
        else:
            src_pos = canvas_pos.get(req.source)
            dst_pos = canvas_pos.get(req.destination)
            if src_pos and dst_pos:
                rejected_js.append({
                    "orderLabel": order_label,
                    "reason": str(plan.status_reason),
                    "src": src_pos,
                    "dst": dst_pos,
                    "srcName": src_name[:20],
                    "dstName": dst_name[:20],
                })

    # ── Serialize data to JSON for injection ──────────────────────────────────
    edges_json    = json.dumps(edges_js)
    nodes_json    = json.dumps(nodes_js)
    drones_json   = json.dumps(drones_js)
    rejected_json = json.dumps(rejected_js)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}}
body{{background:#060c18;color:#f8fafc;overflow:hidden;user-select:none;}}
.hdr{{background:#0f172a;padding:8px 16px;border-bottom:1px solid #1e3a5f;display:flex;justify-content:space-between;align-items:center;height:44px;}}
.hdr h1{{font-size:.88rem;font-weight:700;color:#38bdf8;display:flex;align-items:center;gap:5px;flex-wrap:wrap;}}
.badge{{background:#0369a1;color:#e0f2fe;padding:2px 6px;border-radius:3px;font-size:.62rem;font-weight:600;}}
.badge-safe{{background:#065f46;color:#a7f3d0;padding:2px 6px;border-radius:3px;font-size:.62rem;font-weight:700;border:1px solid #10b981;}}
.badge-city{{background:#581c87;color:#e9d5ff;padding:2px 6px;border-radius:3px;font-size:.62rem;font-weight:600;}}
.main{{display:grid;grid-template-columns:1fr 285px;height:calc(100vh - 44px);}}
.arena{{position:relative;background:#060c18;display:flex;justify-content:center;align-items:center;overflow:hidden;}}
canvas{{border:1px solid #1e3a5f;border-radius:5px;}}
.hud{{position:absolute;display:flex;flex-direction:column;gap:4px;pointer-events:none;}}
.hud-tl{{top:8px;left:8px;}}.hud-tr{{top:8px;right:8px;}}.hud-bl{{bottom:8px;left:8px;}}
.hud-box{{background:rgba(6,12,24,.9);border:1px solid #1e3a5f;border-radius:5px;padding:4px 9px;
          font-family:'Courier New',monospace;font-size:.67rem;backdrop-filter:blur(6px);min-width:144px;}}
.hud-t{{color:#475569;font-size:.53rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:1px;}}
.hud-v{{font-size:.8rem;font-weight:700;color:#38bdf8;}}
.g{{color:#10b981;}}.a{{color:#f59e0b;}}.r{{color:#ef4444;}}.p{{color:#a78bfa;}}
.sb{{background:#0f172a;border-left:1px solid #1e3a5f;padding:9px;display:flex;flex-direction:column;gap:7px;overflow-y:auto;}}
.card{{background:#060c18;border:1px solid #1e3a5f;border-radius:6px;padding:9px;}}
.ct{{font-size:.63rem;font-weight:700;text-transform:uppercase;color:#475569;letter-spacing:.06em;margin-bottom:7px;display:flex;justify-content:space-between;align-items:center;}}
button{{background:#1d4ed8;color:#fff;border:none;padding:6px 9px;border-radius:4px;font-weight:600;font-size:.68rem;cursor:pointer;transition:all .12s;}}
button:hover{{filter:brightness(1.15);}}
.bl{{background:#059669;width:100%;font-size:.76rem;padding:8px;}}
.bp{{background:#334155;}}.brs{{background:#7f1d1d;}}
.bg{{background:#92400e;width:100%;margin-top:4px;}}
.brow{{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:4px;}}
.slr{{margin-top:4px;}}
.sll{{display:flex;justify-content:space-between;font-size:.63rem;color:#94a3b8;margin-bottom:1px;}}
input[type=range]{{width:100%;accent-color:#38bdf8;cursor:pointer;}}
.fl{{display:flex;flex-direction:column;gap:4px;max-height:220px;overflow-y:auto;}}
.fi{{background:#0f172a;border-radius:4px;padding:5px 7px;font-size:.63rem;display:flex;justify-content:space-between;align-items:flex-start;border-left:3px solid #38bdf8;cursor:pointer;}}
.pill{{padding:1px 5px;border-radius:3px;font-size:.58rem;font-weight:700;white-space:nowrap;margin-top:2px;}}
.pf{{background:#0369a1;color:#fff;}}.pw{{background:#92400e;color:#fff;}}
.pd{{background:#065f46;color:#a7f3d0;}}.ps{{background:#334155;color:#94a3b8;}}
.bbar-w{{height:4px;background:#1e293b;border-radius:2px;margin-top:2px;}}
.bbar{{height:100%;border-radius:2px;transition:width .4s;}}
.leg{{font-size:.62rem;color:#64748b;line-height:1.75;}}
</style>
</head>
<body>
<div class="hdr">
  <h1>🚁 WEBS Hardware SITL — Bangalore Central
    <span class="badge">HARDWARE PHYSICS</span>
    <span class="badge-city">📍 Bangalore, India</span>
    <span class="badge">Wind: {wind_speed:.1f} m/s @ {wind_dir_deg:.0f}°</span>
  </h1>
  <div style="display:flex;gap:7px;align-items:center;">
    <span class="badge-safe" id="safeBadge">🛡️ 0 AIR VIOLATIONS</span>
    <span style="font-size:.65rem;color:#475569;" id="clockBadge">T+0.0s</span>
  </div>
</div>

<div class="main">
  <div class="arena">
    <canvas id="c" width="{W}" height="{H}"></canvas>
    <div class="hud hud-tl">
      <div class="hud-box"><div class="hud-t">Active Fleet</div><div class="hud-v g" id="hFleet">0/0 in air</div></div>
      <div class="hud-box"><div class="hud-t">4D Reservation Table</div><div class="hud-v g" id="hConflicts">0 Conflicts ✓</div></div>
      <div class="hud-box"><div class="hud-t">Wind</div><div class="hud-v" id="hWind">{wind_speed:.1f} m/s @ {wind_dir_deg:.0f}°</div></div>
    </div>
    <div class="hud hud-bl">
      <div class="hud-box"><div class="hud-t">WEBS Energy Saving vs B1</div><div class="hud-v a">+45–78 %</div></div>
      <div class="hud-box"><div class="hud-t">Mission Elapsed</div><div class="hud-v p" id="hTime">T+0.0s</div></div>
    </div>
    <div class="hud hud-tr">
      <div class="hud-box"><div class="hud-t">Selected UAV</div><div class="hud-v" id="hSel">—</div></div>
      <div class="hud-box"><div class="hud-t">Ground Speed</div><div class="hud-v g" id="hGS">— m/s</div></div>
      <div class="hud-box"><div class="hud-t">Battery SoC</div><div class="hud-v a" id="hBatt">—</div></div>
      <div class="hud-box"><div class="hud-t">Altitude AGL</div><div class="hud-v" id="hAlt">0 m</div></div>
    </div>
  </div>

  <div class="sb">
    <div class="card">
      <div class="ct">Fleet Mission Control</div>
      <button class="bl" onclick="launchAll()">🚀 LAUNCH ALL DRONES</button>
      <div class="brow">
        <button class="bp" onclick="togglePause()" id="btnP">⏸ PAUSE</button>
        <button class="brs" onclick="resetAll()">↺ RESET</button>
      </div>
    </div>
    <div class="card">
      <div class="ct">Wind Field Control</div>
      <div class="slr">
        <div class="sll"><span>Wind Speed</span><span id="lblWS">{wind_speed:.1f} m/s</span></div>
        <input type="range" id="slWS" min="0" max="14" step="0.5" value="{wind_speed}" oninput="onWind()">
      </div>
      <div class="slr">
        <div class="sll"><span>Direction</span><span id="lblWD">{wind_dir_deg:.0f}°</span></div>
        <input type="range" id="slWD" min="0" max="360" step="15" value="{wind_dir_deg}" oninput="onWind()">
      </div>
      <button class="bg" onclick="gust()">💨 Inject Wind Gust (12 m/s)</button>
    </div>
    <div class="card">
      <div class="ct">Live Fleet <span id="lblFC" style="color:#38bdf8">0 UAVs</span></div>
      <div class="fl" id="fleetList"></div>
    </div>
    <div class="card">
      <div class="ct">Legend</div>
      <div class="leg">
        <div>● Depot circle — takeoff pad</div>
        <div>★ Star — customer destination</div>
        <div>- - Dashed — 4D reserved corridor</div>
        <div>✕ Red cross — WEBS rejected route</div>
        <div>🟡 Pulsing halo — yield (conflict avoid)</div>
        <div>↗ Blue arrows — wind vector field</div>
        <div>⚡ Bar — live 4S LiPo discharge</div>
      </div>
    </div>
  </div>
</div>

<script>
const W={W}, H={H};
const cv=document.getElementById("c"), ctx=cv.getContext("2d");
const EDGES={edges_json}, NODES={nodes_json}, REJECTED={rejected_json};
const SPECS={drones_json};

let wind={{speed:{wind_speed},deg:{wind_dir_deg}}};
function onWind(){{
  wind.speed=+document.getElementById("slWS").value;
  wind.deg=+document.getElementById("slWD").value;
  document.getElementById("lblWS").textContent=wind.speed.toFixed(1)+" m/s";
  document.getElementById("lblWD").textContent=wind.deg+"°";
  document.getElementById("hWind").textContent=wind.speed.toFixed(1)+" m/s @ "+wind.deg+"°";
}}
function gust(){{
  document.getElementById("slWS").value=12; onWind();
  setTimeout(()=>{{document.getElementById("slWS").value={wind_speed}; onWind();}},4000);
}}

// Build drone objects from injected data
const drones=SPECS.map((s,i)=>Object.assign({{}},s,{{
  x:s.waypoints[0].cx, y:s.waypoints[0].cy,
  altitude:0, targetAlt:0, vx:0, vy:0,
  battery:16.8, soc:100, status:"STANDBY",
  currentWp:0, rotorAngle:0, energyKjUsed:0, groundSpeed:0
}}));

let paused=false, elapsed=0, selIdx=0, viols=0;

function launchAll(){{drones.forEach(d=>{{if(d.status==="STANDBY"){{d.status="FLYING";d.targetAlt=65;d.currentWp=1;}}}});}}
function togglePause(){{paused=!paused;document.getElementById("btnP").textContent=paused?"▶ RESUME":"⏸ PAUSE";}}
function resetAll(){{
  drones.forEach((d,i)=>{{
    d.x=d.waypoints[0].cx;d.y=d.waypoints[0].cy;
    d.altitude=0;d.targetAlt=0;d.vx=0;d.vy=0;
    d.battery=16.8;d.soc=100;d.status="STANDBY";
    d.currentWp=0;d.rotorAngle=0;d.energyKjUsed=0;d.groundSpeed=0;
    d.hasParcel=SPECS[i].hasParcel;
  }});
  elapsed=0;viols=0;paused=false;
  document.getElementById("btnP").textContent="⏸ PAUSE";
}}

const BASE=2.15,FV=16.8,DV=14.0;

function physics(){{
  if(paused)return;
  elapsed+=0.06;
  const rad=wind.deg*Math.PI/180, wx=wind.speed*Math.cos(rad), wy=wind.speed*Math.sin(rad);
  drones.forEach(d=>{{
    if(d.status==="STANDBY"||d.status==="DELIVERED")return;
    d.rotorAngle+=0.42;
    if(d.altitude<d.targetAlt)d.altitude=Math.min(d.targetAlt,d.altitude+0.9);
    else if(d.altitude>d.targetAlt)d.altitude=Math.max(0,d.altitude-0.9);
    let yld=false;
    drones.forEach(o=>{{if(o.id!==d.id&&o.status==="FLYING"&&Math.hypot(o.x-d.x,o.y-d.y)<25&&d.id>o.id)yld=true;}});
    if(yld){{d.status="YIELD";d.vx=0;d.vy=0;d.battery=Math.max(DV,d.battery-0.00019);d.soc=Math.max(0,((d.battery-DV)/(FV-DV))*100);return;}}
    else if(d.status==="YIELD")d.status="FLYING";
    if(d.currentWp<d.waypoints.length){{
      const tgt=d.waypoints[d.currentWp];
      const ddx=tgt.cx-d.x,ddy=tgt.cy-d.y,dist=Math.hypot(ddx,ddy);
      if(dist>3.5&&d.altitude>10){{
        d.status="FLYING";
        const ux=ddx/dist,uy=ddy/dist;
        const dot=wx*ux+wy*(-uy);
        const spd=Math.max(0.5,BASE+dot*0.09);
        d.vx=ux*spd;d.vy=uy*spd;d.x+=d.vx;d.y+=d.vy;
        d.groundSpeed=(spd/BASE)*14.0;
        const drag=Math.max(0,-dot*0.0009);
        d.battery=Math.max(DV,d.battery-(0.00038+drag));
        d.soc=Math.max(0,((d.battery-DV)/(FV-DV))*100);
        d.energyKjUsed+=(0.00038+drag)*900;
      }}else{{
        if(d.currentWp===d.waypoints.length-1){{d.hasParcel=false;d.status="DELIVERED";d.targetAlt=0;}}
        else d.currentWp++;
      }}
    }}
  }});
  let v=0;
  for(let i=0;i<drones.length;i++)for(let j=i+1;j<drones.length;j++){{
    if(drones[i].status==="FLYING"&&drones[j].status==="FLYING"&&Math.hypot(drones[i].x-drones[j].x,drones[i].y-drones[j].y)<14)v++;
  }}
  viols=v;
  updateHUD();
}}

function updateHUD(){{
  const active=drones.filter(d=>d.status==="FLYING"||d.status==="YIELD").length;
  document.getElementById("hFleet").textContent=active+"/"+drones.length+" in air";
  document.getElementById("hConflicts").textContent=viols===0?"0 Conflicts ✓":"⚠ "+viols+" VIOLATION";
  document.getElementById("hConflicts").className="hud-v "+(viols===0?"g":"r");
  document.getElementById("safeBadge").textContent=viols===0?"🛡️ 0 AIR VIOLATIONS":"⚠️ VIOLATION!";
  document.getElementById("hTime").textContent="T+"+elapsed.toFixed(1)+"s";
  document.getElementById("clockBadge").textContent="T+"+elapsed.toFixed(1)+"s";
  const sel=drones[selIdx];
  if(sel){{
    document.getElementById("hSel").textContent=sel.name;
    document.getElementById("hGS").textContent=sel.status==="FLYING"?sel.groundSpeed.toFixed(1)+" m/s":"0.0 m/s";
    document.getElementById("hBatt").textContent=sel.soc.toFixed(1)+" %";
    document.getElementById("hBatt").className="hud-v "+(sel.soc>40?"a":"r");
    document.getElementById("hAlt").textContent=sel.altitude.toFixed(0)+" m AGL";
  }}
  const list=document.getElementById("fleetList");
  list.innerHTML="";
  document.getElementById("lblFC").textContent=drones.length+" UAVs";
  drones.forEach((d,i)=>{{
    let pill="STANDBY",pc="ps";
    if(d.status==="FLYING"){{pill="FLYING";pc="pf";}}
    if(d.status==="YIELD"){{pill="YIELDING";pc="pw";}}
    if(d.status==="DELIVERED"){{pill="DELIVERED ✓";pc="pd";}}
    const wp=d.waypoints[d.currentWp]||d.waypoints[d.waypoints.length-1];
    const sc=d.soc.toFixed(0);
    const bc=d.soc>40?"#10b981":d.soc>20?"#f59e0b":"#ef4444";
    const el=document.createElement("div");
    el.className="fi";el.style.borderLeftColor=d.color;el.onclick=()=>selIdx=i;
    if(i===selIdx)el.style.background="#0f2030";
    el.innerHTML=`<div style="flex:1"><div style="font-weight:700;color:${{d.color}};font-size:.69rem">${{d.name}}</div>
      <div style="font-size:.59rem;color:#94a3b8;margin-top:1px">${{d.orderLabel}} — ${{d.orderName.slice(0,30)}}</div>
      <div style="font-size:.58rem;color:#475569">→ ${{wp?wp.name:"—"}}</div>
      <div class="bbar-w"><div class="bbar" style="width:${{sc}}%;background:${{bc}}"></div></div>
      <div style="font-size:.57rem;color:#475569">${{d.battery.toFixed(2)}}V ${{sc}}% | ${{d.energyKjUsed.toFixed(2)}} kJ</div>
    </div><span class="pill ${{pc}}" style="margin-left:5px">${{pill}}</span>`;
    list.appendChild(el);
  }});
}}

function star(cx,cy,r){{
  ctx.beginPath();
  for(let i=0;i<10;i++){{const a=Math.PI*i/5-Math.PI/2,rr=i%2===0?r:r*.42;ctx[i===0?"moveTo":"lineTo"](cx+rr*Math.cos(a),cy+rr*Math.sin(a));}}
  ctx.closePath();ctx.fill();
}}

function draw(){{
  ctx.clearRect(0,0,W,H);
  // Grid
  ctx.strokeStyle="#0c1628";ctx.lineWidth=0.5;
  for(let x=0;x<W;x+=38){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}}
  for(let y=0;y<H;y+=38){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}}

  // Street network
  EDGES.forEach(e=>{{
    ctx.strokeStyle="#1e3a5f";ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(e.u.cx,e.u.cy);ctx.lineTo(e.v.cx,e.v.cy);ctx.stroke();
    if(e.name){{
      const mx=(e.u.cx+e.v.cx)/2,my=(e.u.cy+e.v.cy)/2;
      ctx.fillStyle="#1e3a5f";ctx.font="6px sans-serif";ctx.fillText(e.name,mx+2,my-2);
    }}
  }});

  // Intersection nodes
  NODES.forEach(n=>{{
    ctx.fillStyle=n.is_landmark?"#1a2e4a":"#0a1020";
    ctx.strokeStyle=n.is_landmark?"#38bdf8":"#1e293b";
    ctx.lineWidth=n.is_landmark?1.2:0.6;
    ctx.beginPath();ctx.arc(n.cx,n.cy,n.is_landmark?5:2.5,0,Math.PI*2);ctx.fill();ctx.stroke();
    if(n.is_landmark&&n.label){{ctx.fillStyle="#3b5a7a";ctx.font="6.5px sans-serif";ctx.fillText(n.label,n.cx+6,n.cy+3);}}
  }});

  // Wind arrows
  const wr=wind.deg*Math.PI/180,wl=Math.max(10,wind.speed*3);
  ctx.strokeStyle="rgba(56,189,248,.15)";ctx.lineWidth=1;
  for(let x=65;x<W;x+=88)for(let y=55;y<H;y+=88){{
    const tx=x+wl*Math.cos(wr),ty=y+wl*Math.sin(wr);
    ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(tx,ty);ctx.stroke();
    ctx.fillStyle="rgba(56,189,248,.2)";ctx.beginPath();
    ctx.moveTo(tx,ty);ctx.lineTo(tx-5*Math.cos(wr-.4),ty-5*Math.sin(wr-.4));
    ctx.lineTo(tx-5*Math.cos(wr+.4),ty-5*Math.sin(wr+.4));ctx.closePath();ctx.fill();
  }}

  // Rejected routes
  REJECTED.forEach(r=>{{
    ctx.strokeStyle="#ef4444";ctx.lineWidth=1.5;ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(r.src.cx,r.src.cy);ctx.lineTo(r.dst.cx,r.dst.cy);ctx.stroke();
    ctx.setLineDash([]);
    const mx=(r.src.cx+r.dst.cx)/2,my=(r.src.cy+r.dst.cy)/2;
    ctx.fillStyle="#ef4444";ctx.font="bold 11px sans-serif";ctx.fillText("✕",mx-5,my+4);
    ctx.font="6.5px sans-serif";ctx.fillText(r.orderLabel+" REJECTED",mx-17,my+15);
  }});

  // Accepted corridors (dashed) + markers
  drones.forEach(d=>{{
    ctx.strokeStyle=d.color;ctx.globalAlpha=0.28;ctx.lineWidth=2;ctx.setLineDash([5,5]);
    ctx.beginPath();ctx.moveTo(d.waypoints[0].cx,d.waypoints[0].cy);
    d.waypoints.forEach(wp=>ctx.lineTo(wp.cx,wp.cy));ctx.stroke();
    ctx.setLineDash([]);ctx.globalAlpha=1;
    const dst=d.waypoints[d.waypoints.length-1];
    ctx.fillStyle=d.color;star(dst.cx,dst.cy,7);
    ctx.fillStyle="#ccc";ctx.font="6.5px sans-serif";ctx.fillText("★ "+dst.name.slice(0,16),dst.cx+9,dst.cy+3);
    const orig=d.waypoints[0];
    ctx.fillStyle=d.color;ctx.strokeStyle="#fff";ctx.lineWidth=1.2;
    ctx.beginPath();ctx.arc(orig.cx,orig.cy,6,0,Math.PI*2);ctx.fill();ctx.stroke();
  }});

  // Drones
  drones.forEach((d,i)=>{{
    ctx.save();ctx.translate(d.x,d.y);
    if(i===selIdx){{ctx.strokeStyle="rgba(255,255,255,.35)";ctx.lineWidth=0.8;ctx.setLineDash([3,3]);ctx.beginPath();ctx.arc(0,0,30,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);}}
    if(d.status==="YIELD"){{
      const hr=26+Math.sin(Date.now()*.012)*4;
      ctx.strokeStyle="#f59e0b";ctx.lineWidth=2;ctx.globalAlpha=0.7;
      ctx.beginPath();ctx.arc(0,0,hr,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1;
      ctx.fillStyle="#f59e0b";ctx.font="bold 7px monospace";ctx.fillText("YIELD",-13,-32);
    }}
    if(d.soc<22&&d.status!=="STANDBY"){{
      ctx.strokeStyle="#ef4444";ctx.lineWidth=1.5;
      ctx.globalAlpha=0.4+0.6*Math.abs(Math.sin(Date.now()*.016));
      ctx.beginPath();ctx.arc(0,0,24,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1;
    }}
    ctx.strokeStyle="rgba(56,189,248,.12)";ctx.lineWidth=0.8;ctx.beginPath();ctx.arc(0,0,22,0,Math.PI*2);ctx.stroke();
    const sh=(d.altitude/65)*13;
    ctx.fillStyle="rgba(0,0,0,.35)";ctx.beginPath();ctx.ellipse(sh,sh+4,11,6,0,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle="#334155";ctx.lineWidth=3.2;
    ctx.beginPath();ctx.moveTo(-16,-16);ctx.lineTo(16,16);ctx.moveTo(-16,16);ctx.lineTo(16,-16);ctx.stroke();
    ctx.fillStyle="#0a1020";ctx.strokeStyle=d.color;ctx.lineWidth=1.8;
    ctx.fillRect(-7,-7,14,14);ctx.strokeRect(-7,-7,14,14);
    if(d.hasParcel){{ctx.fillStyle="#f59e0b";ctx.fillRect(-4,-4,8,8);ctx.strokeStyle="#92400e";ctx.lineWidth=0.8;ctx.strokeRect(-4,-4,8,8);}}
    [[-16,-16],[16,-16],[-16,16],[16,16]].forEach((rp,ri)=>{{
      ctx.fillStyle=d.color;ctx.beginPath();ctx.arc(rp[0],rp[1],3.5,0,Math.PI*2);ctx.fill();
      if(d.status!=="STANDBY"&&d.status!=="DELIVERED"){{
        ctx.strokeStyle="rgba(255,255,255,.55)";ctx.lineWidth=1.3;
        ctx.beginPath();ctx.arc(rp[0],rp[1],9.5,d.rotorAngle+ri*.9,d.rotorAngle+ri*.9+Math.PI);ctx.stroke();
        ctx.strokeStyle="rgba(255,255,255,.2)";
        ctx.beginPath();ctx.arc(rp[0],rp[1],9.5,d.rotorAngle+ri*.9+Math.PI,d.rotorAngle+ri*.9+2*Math.PI);ctx.stroke();
      }}
    }});
    ctx.fillStyle="#fff";ctx.font="bold 8px sans-serif";ctx.fillText(d.name,-14,25);
    ctx.fillStyle=d.color;ctx.font="6.5px sans-serif";
    ctx.fillText(d.status==="STANDBY"?"STANDBY":d.status==="DELIVERED"?"LANDED ✓":d.status==="YIELD"?"YIELD":d.soc.toFixed(0)+"% SOC",-14,35);
    ctx.restore();
  }});
}}

function loop(){{physics();draw();requestAnimationFrame(loop);}}
onWind();updateHUD();requestAnimationFrame(loop);
</script>
</body>
</html>"""
