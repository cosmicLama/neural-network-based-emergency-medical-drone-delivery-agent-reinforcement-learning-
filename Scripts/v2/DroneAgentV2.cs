using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

// v2 of the drone agent. Identical flight/stability setup to the v1 DroneAgent,
// but it also has to fly to a randomly placed target. Each time the target is
// reached it is moved to a new point and the agent keeps chasing an endless
// sequence of targets within a single episode.
//
// Reward design notes (after the first round of training collapsed into a
// "balance and slowly sink" hover):
//   - The per-step upright reward is kept tiny and is smaller than the time
//     penalty, so simply hovering bleeds reward instead of farming it.
//   - The angular-velocity penalty is light so it does not punish the tilt that
//     a multicopter MUST do to translate toward the target.
//   - A dense velocity-alignment reward gives an immediate "fly that way" signal
//     to bootstrap movement, on top of the ratcheted progress reward.
//   - A simple staged spawn keeps the first target very close, the second a bit
//     further, then opens up to the full random box.
public class DroneAgentV2 : Agent
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
    // How close the drone has to get before the target counts as reached.
    [SerializeField]
    private float reachDistance = 3f;

    [Header("Reward Weights")]
    // Progress reward per unit of NEW closest-distance to the target. Only the best
    // (smallest) distance reached so far is ever paid out, so the drone cannot farm
    // reward by oscillating toward/away from the target.
    [SerializeField]
    private float progressRewardScale = 1f;
    // Dense shaping: reward velocity that points at the target. Symmetric (flying
    // away is penalised by the same amount), so it cannot be farmed, but it gives an
    // immediate gradient that rewards tilting toward the target right away.
    [SerializeField]
    private float velocityAlignmentScale = 0.01f;
    // Per-step cost. Kept LARGER than the upright reward so that just hovering is a
    // net loss and the agent is pushed to actually reach targets for the +reach bonus.
    [SerializeField]
    private float timePenalty = 0.01f;
    // One-off bonus granted when a target is reached.
    [SerializeField]
    private float reachReward = 10f;
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

    private Bounds bounds;
    private Resetter resetter;

    private Transform target;
    private Vector3 boundsCenter;

    // Closest the drone has gotten to the current target so far. Progress reward is
    // only granted when this record is beaten, then it ratchets down.
    private float bestDistance;

    // How many targets have been reached in the current episode. Drives the simple
    // staged spawn (near -> mid -> full random).
    private int targetsReachedThisEpisode;

    private StatsRecorder stats;

    public override void Initialize()
    {
        multicopter.Initialize();
        boundsCenter = transform.position;
        bounds = new Bounds(boundsCenter, Vector3.one * 100f);
        resetter = new Resetter(transform);
        stats = Academy.Instance.StatsRecorder;

        CreateTarget();
    }

    public override void OnEpisodeBegin()
    {
        // Report how the previous episode went before resetting the counter.
        stats?.Add("DroneV2/TargetsReached", targetsReachedThisEpisode);

        resetter.Reset();
        targetsReachedThisEpisode = 0;
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
        multicopter.UpdateThrust(actionBuffers.ContinuousActions.Array);

        if (bounds.Contains(multicopter.Frame.position))
        {
            Rigidbody body = multicopter.Rigidbody;

            // Urgency: every step that hasn't delivered costs a little.
            AddReward(-timePenalty);

            // Light stability shaping (kept minor so it doesn't dominate the goal).
            AddReward(multicopter.Frame.up.y * uprightRewardScale);
            AddReward(body.angularVelocity.magnitude * -angularVelocityPenalty);
            AddReward(body.velocity.magnitude * -linearVelocityPenalty);

            Vector3 toTarget = target.position - multicopter.Frame.position;
            float distance = toTarget.magnitude;

            // Dense directional signal: reward moving toward the target.
            if (distance > 1e-4f)
            {
                Vector3 dirToTarget = toTarget / distance;
                AddReward(Vector3.Dot(dirToTarget, body.velocity) * velocityAlignmentScale);
            }

            // Ratcheted progress: only pay for beating the closest distance so far.
            if (distance < bestDistance)
            {
                AddReward((bestDistance - distance) * progressRewardScale);
                bestDistance = distance;
            }

            if (distance < reachDistance)
            {
                AddReward(reachReward);
                targetsReachedThisEpisode++;
                MoveTarget();
            }
        }
        else
        {
            resetter.Reset();
        }
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

        go.transform.localScale = Vector3.one * (reachDistance * 0.5f);
        target = go.transform;
        MoveTarget();
    }

    private void MoveTarget()
    {
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
    }
}
