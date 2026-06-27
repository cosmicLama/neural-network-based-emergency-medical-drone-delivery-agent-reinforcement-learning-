using UnityEngine;

public class GizmoAxes : MonoBehaviour
{
    [SerializeField]
    private Color right = Color.red;
    [SerializeField]
    private Color up = Color.green;
    [SerializeField]
    private Color forward = Color.blue;

    [SerializeField]
    private float length = 1;
    [SerializeField]
    private bool draw = true;

    private void OnDrawGizmos()
    {
        if (!draw)
        {
            return;
        }

        Vector3 origin = transform.position;

        Gizmos.color = right;
        Gizmos.DrawRay(origin, transform.right * length);

        Gizmos.color = up;
        Gizmos.DrawRay(origin, transform.up * length);

        Gizmos.color = forward;
        Gizmos.DrawRay(origin, transform.forward * length);
    }
}
