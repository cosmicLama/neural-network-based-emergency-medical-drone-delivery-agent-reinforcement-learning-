using UnityEngine;

public class Rotor : MonoBehaviour
{
    public float CurrentThrust { get; private set; }

    public bool Reversable { get; set; }
    public float ThrustResponse { get; set; }
    public float ThrustScale { get; set; }
    public float TorqueScale { get; set; }

    [SerializeField]
    private Transform outerRing;
    [SerializeField]
    private Transform innerRing;
    [SerializeField]
    private Transform rotorBlade;

    [SerializeField]
    private float signZ;
    [SerializeField]
    private float signX;

    private const float bladeSpinSpeed = 2400f;

    private Rigidbody innerBody;
    private ConfigurableJoint outerJoint;
    private ConfigurableJoint innerJoint;
    private float spinSign;

    public void Initialize()
    {
        innerBody = innerRing.GetComponent<Rigidbody>();
        outerJoint = outerRing.GetComponent<ConfigurableJoint>();
        innerJoint = innerRing.GetComponent<ConfigurableJoint>();
        spinSign = rotorBlade.name == "RotorCW" ? 1f : -1f;
    }

    public void OnReset()
    {
        CurrentThrust = 0f;
    }

    public void UpdateThrust(float thrustNorm, float deltaTime)
    {
        if (!Reversable)
        {
            thrustNorm = (thrustNorm + 1f) * 0.5f;
        }

        CurrentThrust = Mathf.Lerp(CurrentThrust, thrustNorm, deltaTime * ThrustResponse);

        Vector3 axis = innerRing.up;
        innerBody.AddForce(axis * (CurrentThrust * ThrustScale), ForceMode.Impulse);
        innerBody.AddRelativeTorque(axis * (CurrentThrust * TorqueScale * -spinSign), ForceMode.Impulse);
    }

    public void UpdateTilt(Quaternion rot, float yawAngle)
    {
        Vector3 euler = (Quaternion.Inverse(rot) * transform.localRotation).eulerAngles;
        innerJoint.targetRotation = Quaternion.Euler(euler.x + yawAngle * signX, 0f, 0f);
        outerJoint.targetRotation = Quaternion.Euler(0f, 0f, euler.z + yawAngle * signZ);
    }

    private void Update()
    {
        float spin = CurrentThrust * bladeSpinSpeed * spinSign * Time.deltaTime;
        rotorBlade.Rotate(0f, spin, 0f, Space.Self);
    }
}
