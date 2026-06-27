using System.Collections.Generic;
using UnityEngine;

public class ResettableItem
{
    private readonly Transform target;
    private readonly Rigidbody body;
    private readonly ConfigurableJoint joint;

    private readonly Vector3 startLocalPosition;
    private readonly Quaternion startLocalRotation;

    public ResettableItem(Transform tf)
    {
        target = tf;
        startLocalPosition = tf.localPosition;
        startLocalRotation = tf.localRotation;
        body = tf.GetComponent<Rigidbody>();
        joint = tf.GetComponent<ConfigurableJoint>();
    }

    public void Reset()
    {
        if (body != null)
        {
            body.velocity = Vector3.zero;
            body.angularVelocity = Vector3.zero;
            body.Sleep();
        }

        if (joint != null)
        {
            joint.targetRotation = Quaternion.identity;
        }

        target.localPosition = startLocalPosition;
        target.localRotation = startLocalRotation;
    }
}

public class Resetter
{
    private readonly List<ResettableItem> items = new List<ResettableItem>();

    public Resetter(Transform root)
    {
        Collect(root);
    }

    public void Reset()
    {
        for (int i = 0; i < items.Count; i++)
        {
            items[i].Reset();
        }
    }

    private void Collect(Transform tf)
    {
        items.Add(new ResettableItem(tf));

        int count = tf.childCount;
        for (int i = 0; i < count; i++)
        {
            Collect(tf.GetChild(i));
        }
    }
}
