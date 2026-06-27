<img width="1097" height="596" alt="Screenshot 2026-06-27 194439" src="https://github.com/user-attachments/assets/4c46fc87-3da4-49d3-bc3a-50167e34e968" />

A simulated multicopter that teaches itself to fly, navigate, and land precisely on a target using deep reinforcement learning (Unity ML-Agents / PPO), built toward autonomous delivery of urgent medical supplies — syringes, medication, first-aid kits — to homes spread across a wide, low-density desert community like Al Qua'a. 

Please Note github is not allowing me to upload my project .unitypackage of 337mb. The project is an autonomous neural network agent and file size 337mb is the most compression i could get. So uploaded scripts which makes the agent work.

# Autonomous Medical Delivery Drone — Reinforcement-Learning Flight Controller

**Tatweer Hackathon 2026 · Challenge 2 — Reaching people quickly across a dispersed community**

A simulated multicopter that teaches itself to fly, navigate, and land precisely on a target using deep reinforcement learning (Unity ML-Agents / PPO), built toward autonomous delivery of urgent medical supplies — syringes, medication, first-aid kits — to homes spread across a wide, low-density desert community like Al Qua'a.

---

## 1. The problem

In Al Qua'a and similar rural communities, homes and camel farms are spread across a large, dispersed area. When someone urgently needs a medical item — an EpiPen, insulin, a syringe, a small medication refill — the time it takes for that item to physically reach them is dictated by road distance and whoever is available to drive it. In an emergency, minutes matter, and distance works against speed.

**Who it's for:** residents of dispersed rural communities who need urgent medical items delivered fast, and the local clinics or pharmacies trying to get those items to them.

**Our approach:** a small autonomous drone that can fly itself directly from a dispatch point to a precise GPS location, hover, and deliver — skipping the road network entirely.

---

## 2. What we built

Rather than hand-coding a flight controller (the traditional approach for drones, which requires significant aerospace tuning per airframe), we trained the drone's flight behavior end-to-end using **reinforcement learning** in a physics-simulated environment. The drone has no pre-programmed notion of "how to fly" — it starts as a quadcopter that can't even hover, and learns flight, navigation, and precision landing entirely through trial and error against a reward signal.

This was built and trained in **Unity** using **ML-Agents** (PPO), with the full multicopter physics (rotor thrust, torque, joints) simulated rather than scripted.

### Why simulation-first
Training a real drone to fly by trial and error means crashing real hardware hundreds of thousands of times. Simulation lets the policy fail safely, fast, and in parallel, and the resulting trained network is small enough to run on cheap onboard hardware (a single neural network forward-pass replaces a hand-tuned flight controller).

### Training progression
We trained the agent in stages, each one building on the last:

1. **Stage 1 — Stability.** The drone first learns the basics of staying airborne: keeping itself upright and stable using only direct control of its four rotors. No destination yet — just learning not to flip or crash.
2. **Stage 2 — Navigation.** Once stable, the drone is given a target point in 3D space and rewarded for closing the distance to it. Targets first spawn very close, then progressively farther away, until the drone can be sent anywhere within a large open flight volume and reliably reach it.
3. **Stage 3 — Precision delivery.** The final stage tightens the goal from "get near the target" to "land on it exactly." The drone must arrive within a very small tolerance of the target, then hold a stable hover directly on top of it for a sustained period (simulating the dwell time needed to lower or release a payload) before the mission is considered complete.

At each stage, drifting, wobbling, or overshooting costs reward, while genuine progress toward (and precision on) the target is rewarded — so the policy is pushed toward smooth, efficient, accurate flight rather than just "eventually getting there."

### Delivery-specific behavior
On top of the learned policy, the final delivery agent adds capabilities aimed specifically at the medical-delivery use case:

- **Multi-waypoint missions** — the drone can be given an ordered sequence of delivery points rather than a single destination, advancing automatically once each is reached.
- **Obstacle avoidance / hard-fail detection** — any collision with a tagged obstacle (buildings, terrain, trees) is detected and penalized as a mission failure, and the episode resets, training the policy to route around obstacles rather than through them.
- **Guaranteed final approach.** Because a learned policy can be imprecise in its last few metres, the final descent onto the exact delivery point is handed off from the trained network to a small hand-written stabilizing controller, which gently brakes the drone's velocity and eases it down onto the exact target instead of relying on the network's raw output. This means the *learned* policy handles the hard part — long-range navigation and obstacle avoidance through open, varied terrain — while a simple, predictable control loop guarantees the final touchdown is accurate, which matters when the payload is a medical item.

### Supporting tooling
We also built supporting Unity tooling to make development and validation possible: a collision-avoiding chase camera for recording flight footage, a debug gizmo for visualizing the drone's local orientation axes, and a reset system that restores every moving part of the drone rig to its starting state between training episodes (essential for running tens of thousands of training episodes unattended).

---

## 3. Why this fits Challenge 2

Distance and dispersion are the explicit barrier this challenge names. A drone that flies in a straight line is not subject to road networks, traffic, or driving time — it collapses the "time to reach a dispersed resident" problem directly. Framed for urgent medical items specifically, the same system that delivers a routine refill today is the same system that can deliver an emergency item in the first critical minutes before a human responder can physically arrive.

---

## 4. Feasibility & deployment path

- **Hardware cost.** The trained control policy is a small neural network (a handful of fully-connected layers) that runs comfortably on the kind of flight-controller-grade compute already found in consumer/prosumer drones — no GPU or cloud connection needed onboard.
- **Payload.** Designed around small, light medical items (syringes, EpiPens, small medication packs) rather than heavy cargo, keeping airframe size and cost down.
- **Training cost, not flight cost.** The expensive part (training) happens once, in simulation, on a developer machine — not per-drone and not per-flight.
- **Maintenance.** Standard quadcopter maintenance (batteries, rotors) applies; no exotic hardware is required beyond what's already common in agricultural/delivery drone platforms.
- **Known limitation / next step.** This project was trained and validated in simulation. The next concrete step toward real-world deployment is transferring the trained policy to a real airframe (sim-to-real transfer) and validating against real wind, GPS noise, and battery dynamics — a well-established path in drone RL research, but one we have not yet run on physical hardware.

---

## 5. Scalability

The training approach is airframe-agnostic by design — the same pipeline can retrain a policy for a different drone size, payload weight, or rotor configuration without redesigning a hand-tuned controller from scratch. Because the destination is just a 3D coordinate, scaling from one delivery route to an entire community's worth of homes and clinics is a matter of feeding the system more waypoints, not redesigning the flight logic. The same approach generalizes to any dispersed rural community, not just Al Qua'a.

---

## 6. How to run / verify

1. Open the project in **Unity** (2022.3 LTS or compatible) with the **ML-Agents** package installed.
2. Open the training scene containing the drone rig (`Multicopter` + four `Rotor` units) and the agent component.
3. To watch a **pre-trained** policy fly: assign the trained `.onnx` model to the agent's Behavior Parameters and press Play.
4. To **train from scratch**: run `mlagents-learn <config>.yaml --run-id=<name>`, then press Play in Unity. Training progress (targets reached per episode, etc.) is logged via ML-Agents' StatsRecorder and visible in TensorBoard.
5. mlagents release version 21 used from github
6. Project file size is over 300mb and github is not allowing me to upload the full project file

**Tools used:** Unity, C#, Unity ML-Agents Toolkit (PPO), TensorBoard for training metrics.

---

## 7. Repository contents

| File | Role |
|---|---|
| `Multicopter.cs` | Physical drone rig — rotor thrust mixing, orientation/inclination, frame |
| `Rotor.cs` | Individual rotor physics (thrust response, torque, visual spin) |
| `DroneAgent.cs` | Stage 1 — stability-only training agent |
| `DroneAgentV2.cs` | Stage 2 — adds chasing randomly placed targets across the full flight area |
| `DroneAgentV4.cs` | Final delivery agent — precision arrival + hover, multi-waypoint missions, obstacle-hit detection, hand-off to a stabilizing landing controller for exact touchdown |
| `Normalization.cs` | Observation-normalization helpers fed into the neural network |
| `Resetter.cs` | Restores the drone rig between training episodes |
| `CameraFollow.cs` | Collision-aware chase camera for recording flights |
| `GizmoAxes.cs` | Debug visualization of local orientation axes |

---

## 8. Team

Syed Raza

## 9. License / acknowledgements

Built on the [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). Submitted to Tatweer Hackathon, 26–28 June 2026, Al Qua'a, Al Ain, UAE.
