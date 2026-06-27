using UnityEngine;

namespace Unity.MLAgentsExamples
{
    public class CameraFollow : MonoBehaviour
    {
        [Header("Follow Settings")]
        [Tooltip("The target to follow")] 
        public Transform target;

        [Tooltip("The time it takes to move to the new position")]
        public float smoothingTime = 0.3f;

        [Header("Follow Axes")]
        public bool followX = true;
        public bool followY = false;
        public bool followZ = true;

        [Header("Collision Settings")]
        [Tooltip("Layers that the camera should collide with")]
        public LayerMask collisionLayers = -1;

        [Tooltip("Layers that the camera should ignore for collision")]
        public LayerMask ignoreLayers = 0;

        [Tooltip("Radius for collision detection")]
        public float collisionRadius = 0.5f;

        [Tooltip("Distance to keep from walls when collision occurs")]
        public float wallBuffer = 0.1f;

        private Vector3 m_Offset;
        private Vector3 m_CamVelocity; //Camera's velocity (used by SmoothDamp)
        private LayerMask m_FinalCollisionMask;

        // Use this for initialization
        void Start()
        {
            if (target == null)
                return;

            m_Offset = gameObject.transform.position - target.position;
            
            // Calculate final collision mask by removing ignored layers from collision layers
            m_FinalCollisionMask = collisionLayers & ~ignoreLayers;
        }

        void FixedUpdate()
        {
            if (target == null)
                return;

            var desiredPosition = new Vector3(
                followX ? target.position.x + m_Offset.x : transform.position.x,
                followY ? target.position.y + m_Offset.y : transform.position.y,
                followZ ? target.position.z + m_Offset.z : transform.position.z);

            // Check for collision and adjust position if needed
            Vector3 finalPosition = CheckCollisionAndAdjust(desiredPosition);

            gameObject.transform.position =
                Vector3.SmoothDamp(transform.position, finalPosition, ref m_CamVelocity, smoothingTime, Mathf.Infinity,
                    Time.fixedDeltaTime);
        }

        private Vector3 CheckCollisionAndAdjust(Vector3 desiredPosition)
        {
            Vector3 currentPos = transform.position;
            Vector3 direction = (desiredPosition - currentPos).normalized;
            float distance = Vector3.Distance(currentPos, desiredPosition);

            // If there's no movement, return current position
            if (distance < 0.001f)
                return desiredPosition;

            // Perform sphere cast to check for collisions
            RaycastHit hit;
            if (Physics.SphereCast(currentPos, collisionRadius, direction, out hit, distance, m_FinalCollisionMask))
            {
                // Calculate safe position with buffer from the wall
                Vector3 safePosition = hit.point - direction * (collisionRadius + wallBuffer);
                
                // Make sure we don't go behind our current position
                if (Vector3.Dot(safePosition - currentPos, direction) > 0)
                {
                    return safePosition;
                }
                else
                {
                    return currentPos;
                }
            }

            return desiredPosition;
        }

        // Update collision mask when values change in inspector (only in editor)
        void OnValidate()
        {
            if (Application.isPlaying)
            {
                m_FinalCollisionMask = collisionLayers & ~ignoreLayers;
            }
        }

        // Draw collision sphere in scene view for debugging
        void OnDrawGizmosSelected()
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(transform.position, collisionRadius);
            
            if (target != null)
            {
                Vector3 desiredPos = new Vector3(
                    followX ? target.position.x + m_Offset.x : transform.position.x,
                    followY ? target.position.y + m_Offset.y : transform.position.y,
                    followZ ? target.position.z + m_Offset.z : transform.position.z);
                Gizmos.color = Color.green;
                Gizmos.DrawLine(transform.position, desiredPos);
            }
        }
    }
}