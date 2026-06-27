using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class DroneAgent : Agent
{
    [SerializeField]
    private Multicopter multicopter;

    private Bounds bounds;
    private Resetter resetter;

    public override void Initialize()
    {
        multicopter.Initialize();
        bounds = new Bounds(transform.position, Vector3.one * 100f);
        resetter = new Resetter(transform);
    }

    public override void OnEpisodeBegin()
    {
        resetter.Reset();
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
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        multicopter.UpdateThrust(actionBuffers.ContinuousActions.Array);

        if (bounds.Contains(multicopter.Frame.position))
        {
            Rigidbody body = multicopter.Rigidbody;
            AddReward(multicopter.Frame.up.y);
            AddReward(body.velocity.magnitude * -0.2f);
            AddReward(body.angularVelocity.magnitude * -0.1f);
        }
        else
        {
            resetter.Reset();
        }
    }
}
