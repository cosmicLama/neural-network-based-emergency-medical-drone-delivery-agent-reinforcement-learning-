using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

// v4 of the drone agent. A duplicate of DroneAgentV2 with two behavioural changes
// requested for the Trainingv4 scene:
//
//   1) The target must be reached "exactly". The arrival tolerance (reachDistance)
//      is shrunk right down so the drone has to sit essentially on top of the
//      target instead of just getting within a few metres. (A literal 0 can never
//      be hit by continuous physics, so a tiny tolerance is used to detect arrival
//      while the hover reward below peaks at distance 0 to drive true precision.)
//
//   2) Reaching the target no longer respawns it immediately. Instead the drone is
//      rewarded for HOVERING right on the target for hoverDuration seconds, and any
//      drift away from the target ("offshoot") is penalised the whole time. Only
//      after the hover window elapses is the target moved to a new location, so the
//      loop becomes: fly to target -> hover precisely -> target respawns -> repeat.
public class DroneAgentV4 : Agent
{
    [SerializeField]
    private Multicopter multicopter;

    [Header("Target")]
    // Half-size of the cube (centered on the agent's start) the target spawns in
    // once the staged warm-up targets have been cleared. The flight bounds use 50,
    // so this is kept smaller to avoid spawning right on the edge.
    [SerializeField]
    private float targetSpawnExtent = 45f;
    // First target of an episode spawns within this radius of the drone (very near).
    [SerializeField]
    private float nearSpawnExtent = 5f;
    // Second target of an episode spawns within this radius of the drone.
    [SerializeField]
    private float midSpawnExtent = 15f;
    // Arrival tolerance. Kept tiny (vs v2's 3) so the drone must reach the target
    // almost exactly before it counts as "reached" and the hover phase begins.
    [SerializeField]
    private float reachDistance = 0.5f;
    // Visual size of the target marker sphere (decoupled from reachDistance so the
    // tiny tolerance does not make the marker invisible).
    [SerializeField]
    private float targetMarkerScale = 1f;
    // When unchecked the generated target sphere keeps existing (and still drives all
    // distance/reward logic) but its mesh is hidden, so the target is invisible in the
    // scene. Toggle it live in the inspector during play.
    [SerializeField]
    private bool showTargetMesh = true;

    [Header("Single Target Mode")]
    // When enabled the target follows a fixed list of waypoints instead of the v2
    // style endless random spawn. The agent flies to each point in order, hovers on
    // it for hoverDuration seconds, then advances to the next; hovering on the LAST
    // point ends the episode.
    [SerializeField]
    private bool singleTargetMode = false;
    // Ordered list of spawn locations for the target. Drag empty GameObjects from the
    // scene here. If left empty the staged random spawn is used as a fallback.
    [SerializeField]
    private Transform[] fixedTargetPoints;

    [Header("Hover")]
    // How long (seconds) the drone must hover on the target before it respawns.
    [SerializeField]
    private float hoverDuration = 5f;
    // Single target mode only: how long (seconds) the drone settles on the LAST
    // waypoint before the episode resets. Kept separate from hoverDuration so the
    // intermediate waypoints can advance instantly (hoverDuration = 0) while the
    // final target still triggers the reset only after this delay. Defaults to 5 so
    // existing scenes that relied on hoverDuration for the final hover are unchanged.
    [SerializeField]
    private float finalResetDelay = 5f;
    // Per-step reward while hovering, scaled by how close to the exact target the
    // drone is (full reward at distance 0, zero at the reach tolerance edge). Over a
    // full hoverDuration this sums to roughly hoverReward * (hoverDuration / fixedDt).
    [SerializeField]
    private float hoverReward = 0.04f;
    // Penalty per unit of distance from the exact target while hovering. This is the
    // "offshoot" penalty: once the target is reached, drifting away in any direction
    // costs reward, so the drone is pushed to stay locked on the target.
    [SerializeField]
    private float offshootPenaltyScale = 0.2f;

    [Header("Reward Weights")]
    // TOTAL progress reward paid out across one approach, regardless of how far the
    // target is. The raw "closest-distance beaten" amount is normalized by the
    // starting distance so a far target and a near target both sum to ~this value
    // (it ratchets, so the drone cannot farm it by oscillating). This keeps the
    // shaping consistent instead of paying far targets much more than near ones.
    [SerializeField]
    private float progressRewardScale = 2f;
    // Dense shaping: reward velocity that points at the target. Symmetric (flying
    // away is penalised by the same amount), so it cannot be farmed, but it gives an
    // immediate gradient that rewards tilting toward the target right away.
    [SerializeField]
    private float velocityAlignmentScale = 0.005f;
    // Per-step cost during the APPROACH phase. Kept LARGER than the upright reward so
    // that just drifting is a net loss and the agent is pushed to actually reach the
    // target for the +reach bonus. Not applied while hovering (hovering is the goal).
    [SerializeField]
    private float timePenalty = 0.005f;
    // One-off bonus granted when a target is reached and the hover phase starts.
    [SerializeField]
    private float reachReward = 5f;
    // Light stability: small reward for staying upright. Deliberately tiny (smaller
    // than the time penalty) so it prevents flipping without dominating the goal.
    [SerializeField]
    private float uprightRewardScale = 0.002f;
    // Light stability: small penalty for spinning. Kept low so it does not punish the
    // tilt needed to fly toward the target.
    [SerializeField]
    private float angularVelocityPenalty = 0.005f;
    // Penalty on raw speed. Off by default because it fights the urgency to fly fast
    // toward the target; exposed in case some damping is wanted.
    [SerializeField]
    private float linearVelocityPenalty = 0f;

    [Header("Observation")]
    // Normalization scale for the relative-target observation (matches the style of
    // the velocity observations rather than shaping behaviour).
    [SerializeField]
    private float targetObservationScale = 0.05f;

    [Header("Obstacle Collision")]
    // Anything carrying this tag counts as an obstacle. Hitting one is a hard fail.
    [SerializeField]
    private string obstacleTag = "obstacle";
    // Penalty applied (as a negative reward) when an obstacle is hit before the
    // episode is reset. Sized to exceed one full target's positive budget so a crash
    // is clearly worse than abandoning a target.
    [SerializeField]
    private float obstaclePenalty = 15f;

    [Header("Manual Landing Override")]
    // Master switch. When on, once the drone gets within landingActivationDistance of
    // the FINAL waypoint (single target mode only) the trained ONNX policy is ignored
    // entirely and a hand-written PD controller flies the drone down onto the exact
    // target and holds it there ("settle & hover"). Turn off to always use the policy.
    [SerializeField]
    private bool manualLandingOverride = true;
    // How close (metres) to the final target the drone must be for the manual landing
    // controller to take over from the policy. Tune this in the inspector.
    [SerializeField]
    private float landingActivationDistance = 10f;
    // Outer position loop: pull toward the target (P) and damp velocity (D). Together
    // they decelerate the drone so it eases onto the target instead of overshooting.
    [SerializeField]
    private float landingPositionP = 2f;
    [SerializeField]
    private float landingVelocityD = 3f;
    // Caps the commanded acceleration so the landing stays gentle/gradual.
    [SerializeField]
    private float landingMaxAcceleration = 8f;
    // Caps how far the drone is allowed to tilt while chasing the target horizontally.
    // Bigger = faster lateral moves but more lift lost to tilt. Keep modest so it does
    // not flip itself toward the ground.
    [SerializeField]
    private float landingMaxTiltDegrees = 25f;
    // Inner attitude loop: tilt the drone so its thrust points along the desired force,
    // then damp the rotation rate so it does not oscillate.
    [SerializeField]
    private float landingAttitudeP = 6f;
    [SerializeField]
    private float landingAttitudeD = 1.5f;
    // Clamp on the per-axis attitude (pitch/roll) command so a big tilt error cannot
    // saturate the rotors and make the drone tumble.
    [SerializeField]
    private float landingMaxAttitudeCommand = 0.5f;
    // Vertical (altitude) hold gains. The throttle self-calibrates: it is seeded from
    // the policy's own rotor output at takeover, then the integral term trims it to
    // whatever actually holds altitude (so it does not depend on mass / thrust scale).
    // P reacts to height error, D damps vertical speed, I removes steady-state droop.
    [SerializeField]
    private float landingThrottleP = 0.25f;
    [SerializeField]
    private float landingThrottleD = 0.3f;
    [SerializeField]
    private float landingThrottleI = 0.5f;
    // Optional fallback hover command used only if the policy's rotor output cannot be
    // sampled at takeover. Normally the auto-seed + integral override this.
    [SerializeField, Range(-1f, 1f)]
    private float landingHoverThrust = 0f;
    // Gravity magnitude used only to set how much tilt a given horizontal demand needs.
    [SerializeField]
    private float landingGravity = 9.81f;

    private Bounds bounds;
    private Resetter resetter;

    private Transform target;
    private MeshRenderer targetRenderer;
    private Vector3 boundsCenter;

    // Reused scratch buffer for the manual landing controller's per-rotor commands so
    // it does not allocate every physics step.
    private float[] manualThrust;
    // Self-calibrating hover throttle (normalized [-1,1]) and whether the manual
    // controller was already running last step (so it seeds the throttle on entry).
    private float hoverThrottle;
    private bool manualWasActive;

    // Closest the drone has gotten to the current target so far. Progress reward is
    // only granted when this record is beaten, then it ratchets down.
    private float bestDistance;

    // Distance to the target when it spawned. Used to normalize the progress reward so
    // every target pays out the same total regardless of how far away it is.
    private float initialDistance;

    // How many targets have been reached in the current episode. Drives the simple
    // staged spawn (near -> mid -> full random).
    private int targetsReachedThisEpisode;

    // Hover state. Once the target is reached the agent enters a hover phase: the
    // target stays put and the drone is rewarded for sitting on it until the timer
    // runs out, after which the target respawns.
    private bool hovering;
    private float hoverTimer;

    // Single target mode: which waypoint in fixedTargetPoints the target is currently on.
    private int targetPointIndex;

    // Guards against multiple drone bodies reporting the same obstacle hit and ending
    // the episode more than once in a single step.
    private bool obstacleHitThisEpisode;

    private StatsRecorder stats;

    public override void Initialize()
    {
        multicopter.Initialize();
        boundsCenter = transform.position;
        bounds = new Bounds(boundsCenter, Vector3.one * 100f);
        // In single target mode the fixed waypoints can sit well outside the default
        // 100-unit flight box. The out-of-bounds guard in OnActionReceived would then
        // force-reset the drone mid-flight on its way to a far waypoint (looks like a
        // random episode reset). Grow the bounds so they contain every waypoint plus a
        // margin to leave room to manoeuvre/settle around it.
        if (singleTargetMode && fixedTargetPoints != null)
        {
            const float boundsMargin = 25f;
            foreach (Transform point in fixedTargetPoints)
            {
                if (point != null)
                {
                    bounds.Encapsulate(point.position + Vector3.one * boundsMargin);
                    bounds.Encapsulate(point.position - Vector3.one * boundsMargin);
                }
            }
        }
        resetter = new Resetter(transform);
        stats = Academy.Instance.StatsRecorder;

        // Collisions are delivered to the GameObject that owns the Rigidbody, NOT to
        // this root object where the agent lives. The drone is a multi-body rig (frame
        // + joint-connected rotors/feet), so any of those bodies could be the one that
        // touches an obstacle. Attach a forwarder to EVERY rigidbody so the hit is
        // caught no matter which part makes contact.
        Rigidbody[] bodies = GetComponentsInChildren<Rigidbody>(true);
        foreach (Rigidbody body in bodies)
        {
            DroneObstacleCollisionForwarder forwarder = body.gameObject.AddComponent<DroneObstacleCollisionForwarder>();
            forwarder.Initialize(this, obstacleTag);
        }
        Debug.Log($"[DroneAgentV4] Obstacle collision forwarders attached to {bodies.Length} rigidbodies, watching tag '{obstacleTag}'.", this);

        CreateTarget();
    }

    public override void OnEpisodeBegin()
    {
        // Report how the previous episode went before resetting the counter.
        stats?.Add("DroneV4/TargetsReached", targetsReachedThisEpisode);

        resetter.Reset();
        targetsReachedThisEpisode = 0;
        hovering = false;
        hoverTimer = 0f;
        targetPointIndex = 0;
        obstacleHitThisEpisode = false;
        manualWasActive = false;
        MoveTarget();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Rigidbody body = multicopter.Rigidbody;

        sensor.AddObservation(multicopter.Inclination);
        sensor.AddObservation(Normalization.Sigmoid(multicopter.LocalizeVector(body.velocity), 0.25f));
        sensor.AddObservation(Normalization.Sigmoid(multicopter.LocalizeVector(body.angularVelocity)));

        Rotor[] rotors = multicopter.Rotors;
        for (int i = 0; i < rotors.Length; i++)
        {
            sensor.AddObservation(rotors[i].CurrentThrust);
        }

        Vector3 toTarget = target.position - multicopter.Frame.position;
        // Compressed relative position (carries rough distance information)...
        sensor.AddObservation(Normalization.Sigmoid(multicopter.LocalizeVector(toTarget), targetObservationScale));
        // ...plus a crisp unit direction that stays informative no matter how far away.
        sensor.AddObservation(multicopter.LocalizeVector(toTarget.normalized));
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Rigidbody body = multicopter.Rigidbody;

        Vector3 toTargetForControl = target.position - multicopter.Frame.position;
        float distanceForControl = toTargetForControl.magnitude;

        // Once we are within range of the FINAL waypoint, ignore the trained policy
        // completely and fly the drone down onto the target with a hand-written PD
        // controller. Otherwise drive the rotors from the ONNX action vector as usual.
        if (ShouldManualLand(distanceForControl))
        {
            ManualLandingControl(body);
        }
        else
        {
            multicopter.UpdateThrust(actionBuffers.ContinuousActions.Array);
            manualWasActive = false;
        }

        if (!bounds.Contains(multicopter.Frame.position))
        {
            resetter.Reset();
            return;
        }

        // Light stability shaping (kept minor so it doesn't dominate the goal). Applied
        // in both phases so the drone stays controllable while approaching and hovering.
        AddReward(multicopter.Frame.up.y * uprightRewardScale);
        AddReward(body.angularVelocity.magnitude * -angularVelocityPenalty);
        AddReward(body.velocity.magnitude * -linearVelocityPenalty);

        Vector3 toTarget = target.position - multicopter.Frame.position;
        float distance = toTarget.magnitude;

        if (hovering)
        {
            HoverStep(distance);
        }
        else
        {
            ApproachStep(body, toTarget, distance);
        }
    }

    // APPROACH phase: fly toward the target. Identical shaping to v2 plus the tighter
    // arrival tolerance that flips the agent into the hover phase.
    private void ApproachStep(Rigidbody body, Vector3 toTarget, float distance)
    {
        // Urgency: every step that hasn't delivered costs a little.
        AddReward(-timePenalty);

        // Dense directional signal: reward moving toward the target.
        if (distance > 1e-4f)
        {
            Vector3 dirToTarget = toTarget / distance;
            AddReward(Vector3.Dot(dirToTarget, body.velocity) * velocityAlignmentScale);
        }

        // Ratcheted progress: only pay for beating the closest distance so far. The
        // gain is normalized by the spawn distance, so the whole approach sums to about
        // progressRewardScale no matter whether the target is near or far.
        if (distance < bestDistance)
        {
            AddReward((bestDistance - distance) / initialDistance * progressRewardScale);
            bestDistance = distance;
        }

        if (distance < reachDistance)
        {
            // Reached the target exactly enough: pay the one-off bonus and switch into
            // the hover phase. The target is NOT moved yet.
            AddReward(reachReward);
            targetsReachedThisEpisode++;
            hovering = true;
            hoverTimer = 0f;
        }
    }

    // HOVER phase: the target stays put. Reward sitting precisely on it and penalise
    // any offshoot (drift away) until the hover timer elapses, then respawn the target.
    private void HoverStep(float distance)
    {
        hoverTimer += Time.fixedDeltaTime;

        // Reward staying close, peaking when the drone is exactly on the target.
        AddReward(hoverReward * Mathf.Clamp01(1f - distance / reachDistance));

        // Offshoot penalty: any distance away from the exact target costs reward, so
        // overshooting or drifting off the target is discouraged the whole hover.
        AddReward(distance * -offshootPenaltyScale);

        if (singleTargetMode)
        {
            if (IsOnFinalWaypoint())
            {
                // ONLY the last (or only) waypoint resets the episode, and only after
                // settling on it for finalResetDelay seconds. Intermediate waypoints
                // never end the episode, so reaching the first target just advances.
                if (hoverTimer >= finalResetDelay)
                {
                    EndEpisode();
                }
            }
            else if (hoverTimer >= hoverDuration)
            {
                // Advance to the next waypoint and hover there next. With
                // hoverDuration = 0 this fires the moment the target is touched, so
                // the episode continues straight on to the next target.
                targetPointIndex++;
                hovering = false;
                hoverTimer = 0f;
                MoveTarget();
            }
        }
        else if (hoverTimer >= hoverDuration)
        {
            hovering = false;
            hoverTimer = 0f;
            MoveTarget();
        }
    }

    // True when the active target is the final (or only) waypoint of a single-target
    // run. Used both for the episode-end timing and the manual landing takeover.
    private bool IsOnFinalWaypoint()
    {
        if (!singleTargetMode)
        {
            return false;
        }

        int count = fixedTargetPoints != null ? fixedTargetPoints.Length : 0;
        return !(count > 1 && targetPointIndex < count - 1);
    }

    // The manual controller takes over only for the final target, only when enabled,
    // and only once the drone is close enough (inside landingActivationDistance).
    private bool ShouldManualLand(float distanceToTarget)
    {
        return manualLandingOverride
            && IsOnFinalWaypoint()
            && distanceToTarget <= landingActivationDistance;
    }

    // Hand-written cascaded PD controller that replaces the ONNX policy for the final
    // approach. It eases the drone onto the exact target position and holds it there:
    //   1) an outer position/velocity loop produces a desired acceleration (clamped so
    //      the descent stays gentle),
    //   2) gravity feed-forward turns that into a desired thrust vector / tilt,
    //   3) an inner attitude loop tilts the airframe to match and damps rotation,
    //   4) the result is mixed onto the individual rotors using their real arm offsets.
    private void ManualLandingControl(Rigidbody body)
    {
        Rotor[] rotors = multicopter.Rotors;
        if (rotors == null || rotors.Length == 0)
        {
            return;
        }

        if (manualThrust == null || manualThrust.Length != rotors.Length)
        {
            manualThrust = new float[rotors.Length];
        }

        float dt = Time.fixedDeltaTime;
        Transform frame = multicopter.Frame;
        Vector3 up = frame.up;

        // On the first manual step, seed the hover throttle from what the policy was
        // already commanding so we start near a real hover instead of free-falling.
        // Rotor.CurrentThrust is the internal [0,1] thrust (non-reversable rig), so map
        // it back to the [-1,1] command space the rotors expect.
        if (!manualWasActive)
        {
            float sum = 0f;
            for (int i = 0; i < rotors.Length; i++)
            {
                sum += rotors[i].CurrentThrust;
            }
            float avgThrust = sum / rotors.Length;
            float seeded = avgThrust * 2f - 1f;
            hoverThrottle = Mathf.Clamp(Mathf.Abs(seeded) > 1e-3f ? seeded : landingHoverThrust, -1f, 1f);
            manualWasActive = true;
        }

        // ---- Vertical: self-calibrating altitude hold (P + D + I) -----------------
        // The integral term trims hoverThrottle to whatever truly holds altitude, so
        // the controller does not depend on knowing the mass or thrust scale.
        float altError = target.position.y - frame.position.y;
        float verticalSpeed = body.velocity.y;
        float pdThrust = landingThrottleP * altError - landingThrottleD * verticalSpeed;
        float baseRaw = hoverThrottle + pdThrust;
        // Anti-windup: only let the integral accumulate while the throttle is NOT
        // saturated, otherwise it winds down during a long descent and then can't
        // recover lift on arrival (the "goes up then slams" symptom).
        if (baseRaw > -1f && baseRaw < 1f)
        {
            hoverThrottle = Mathf.Clamp(hoverThrottle + landingThrottleI * altError * dt, -1f, 1f);
            baseRaw = hoverThrottle + pdThrust;
        }
        // Tilt compensation: when tilted, less thrust points up, so scale the throttle
        // up by 1/cos(tilt) to keep vertical lift roughly constant.
        float cosTilt = Mathf.Clamp(up.y, 0.5f, 1f);
        float baseThrust = Mathf.Clamp(baseRaw / cosTilt, -1f, 1f);

        // ---- Horizontal: desired tilt toward the target (tilt-limited) ------------
        Vector3 posError = target.position - frame.position;
        Vector3 horizErr = new Vector3(posError.x, 0f, posError.z);
        Vector3 horizVel = new Vector3(body.velocity.x, 0f, body.velocity.z);
        Vector3 horizAcc = landingPositionP * horizErr - landingVelocityD * horizVel;
        horizAcc = Vector3.ClampMagnitude(horizAcc, landingMaxAcceleration);
        // Limit how far we are willing to tilt: tan(maxTilt) = |horizAcc| / gravity.
        float maxHorizAcc = landingGravity * Mathf.Tan(landingMaxTiltDegrees * Mathf.Deg2Rad);
        horizAcc = Vector3.ClampMagnitude(horizAcc, maxHorizAcc);

        Vector3 thrustDir = horizAcc + Vector3.up * landingGravity;
        Vector3 desiredUp = thrustDir.sqrMagnitude > 1e-6f ? thrustDir.normalized : Vector3.up;

        // Inner loop: tilt the airframe so its up axis matches desiredUp, damped by the
        // current body-frame angular velocity, and clamp so it cannot saturate/tumble.
        Vector3 attErrorWorld = Vector3.Cross(up, desiredUp); // axis * sin(angle)
        Vector3 attErrorBody = multicopter.LocalizeVector(attErrorWorld);
        Vector3 omegaBody = multicopter.LocalizeVector(body.angularVelocity);
        float pitchCmd = Mathf.Clamp(landingAttitudeP * attErrorBody.x - landingAttitudeD * omegaBody.x, -landingMaxAttitudeCommand, landingMaxAttitudeCommand);
        float rollCmd = Mathf.Clamp(landingAttitudeP * attErrorBody.z - landingAttitudeD * omegaBody.z, -landingMaxAttitudeCommand, landingMaxAttitudeCommand);

        // Normalize the rotor arm offsets so the mixing gains are independent of the
        // physical frame size.
        float maxArm = 1e-3f;
        for (int i = 0; i < rotors.Length; i++)
        {
            Vector3 arm = frame.InverseTransformPoint(rotors[i].transform.position);
            maxArm = Mathf.Max(maxArm, Mathf.Abs(arm.x), Mathf.Abs(arm.z));
        }

        // 4) Mix attitude commands onto each rotor. A rotor forward of center (+z)
        // produces negative pitch torque, one to the right (+x) produces positive roll
        // torque, so allocate thrust accordingly.
        for (int i = 0; i < rotors.Length; i++)
        {
            Vector3 arm = frame.InverseTransformPoint(rotors[i].transform.position);
            float armX = arm.x / maxArm;
            float armZ = arm.z / maxArm;
            float thrust = baseThrust + pitchCmd * (-armZ) + rollCmd * armX;
            manualThrust[i] = Mathf.Clamp(thrust, -1f, 1f);
        }

        multicopter.UpdateThrust(manualThrust);
    }

    private void OnValidate()
    {
        // Allow toggling the target's visibility live in the inspector during play.
        ApplyTargetVisibility();
    }

    // Physics collision path (fires if a collider/rigidbody on this object reports a
    // contact). Kept for completeness alongside the external call below.
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.CompareTag(obstacleTag))
        {
            ApplyObstaclePenalty();
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag(obstacleTag))
        {
            ApplyObstaclePenalty();
        }
    }

    // Hitting an obstacle: hand out the penalty and reset the episode. Public so the
    // forwarder (and any existing raycast/collision detection system) can invoke it.
    public void ApplyObstaclePenalty()
    {
        if (obstacleHitThisEpisode)
        {
            return;
        }

        obstacleHitThisEpisode = true;
        Debug.Log($"[DroneAgentV4] Obstacle hit -> reward {-obstaclePenalty}, ending episode.", this);
        AddReward(-obstaclePenalty);
        EndEpisode();
    }

    private void CreateTarget()
    {
        GameObject go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.name = name + "_Target";

        Collider col = go.GetComponent<Collider>();
        if (col != null)
        {
            Destroy(col);
        }

        go.transform.localScale = Vector3.one * targetMarkerScale;
        target = go.transform;
        targetRenderer = go.GetComponent<MeshRenderer>();
        ApplyTargetVisibility();
        MoveTarget();
    }

    // Shows/hides the target marker's mesh without destroying it, so all the
    // distance/reward logic keeps working while the sphere is invisible.
    private void ApplyTargetVisibility()
    {
        if (targetRenderer != null)
        {
            targetRenderer.enabled = showTargetMesh;
        }
    }

    private void MoveTarget()
    {
        // Single target mode: spawn at the current waypoint and skip the staged
        // random spawn entirely.
        if (singleTargetMode && fixedTargetPoints != null && fixedTargetPoints.Length > 0)
        {
            int i = Mathf.Clamp(targetPointIndex, 0, fixedTargetPoints.Length - 1);
            Transform point = fixedTargetPoints[i];
            if (point != null)
            {
                target.position = point.position;
                bestDistance = Vector3.Distance(multicopter.Frame.position, target.position);
                // Guard against zero so the normalized progress reward never divides by 0.
                initialDistance = Mathf.Max(bestDistance, 1e-4f);
                return;
            }
        }

        // Simple staged spawn: first target very near the drone, second a bit
        // further, third onward anywhere in the full flight box.
        float extent;
        Vector3 origin;
        if (targetsReachedThisEpisode == 0)
        {
            extent = nearSpawnExtent;
            origin = multicopter.Frame.position;
        }
        else if (targetsReachedThisEpisode == 1)
        {
            extent = midSpawnExtent;
            origin = multicopter.Frame.position;
        }
        else
        {
            extent = targetSpawnExtent;
            origin = boundsCenter;
        }

        Vector3 random = new Vector3(
            Random.Range(-extent, extent),
            Random.Range(-extent, extent),
            Random.Range(-extent, extent));

        Vector3 position = origin + random;

        // Keep the target inside the safe flight region regardless of where the
        // drone currently is (the near/mid stages spawn relative to the drone).
        position.x = Mathf.Clamp(position.x, boundsCenter.x - targetSpawnExtent, boundsCenter.x + targetSpawnExtent);
        position.y = Mathf.Clamp(position.y, boundsCenter.y - targetSpawnExtent, boundsCenter.y + targetSpawnExtent);
        position.z = Mathf.Clamp(position.z, boundsCenter.z - targetSpawnExtent, boundsCenter.z + targetSpawnExtent);

        target.position = position;

        // Start the progress ratchet fresh for the new target from where the drone is.
        bestDistance = Vector3.Distance(multicopter.Frame.position, target.position);
        // Guard against zero so the normalized progress reward never divides by 0.
        initialDistance = Mathf.Max(bestDistance, 1e-4f);
    }
}

// Lives on the multicopter's Rigidbody object (added at runtime by the agent) and
// forwards physics collision/trigger events to the agent on the root object, where
// Unity does not deliver them directly.
public class DroneObstacleCollisionForwarder : MonoBehaviour
{
    private DroneAgentV4 agent;
    private string obstacleTag;

    public void Initialize(DroneAgentV4 owner, string tag)
    {
        agent = owner;
        obstacleTag = tag;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (agent != null && collision.collider.CompareTag(obstacleTag))
        {
            Debug.Log($"[DroneAgentV4] Obstacle collision via '{name}' with '{collision.collider.name}'.", this);
            agent.ApplyObstaclePenalty();
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (agent != null && other.CompareTag(obstacleTag))
        {
            Debug.Log($"[DroneAgentV4] Obstacle trigger via '{name}' with '{other.name}'.", this);
            agent.ApplyObstaclePenalty();
        }
    }
}
