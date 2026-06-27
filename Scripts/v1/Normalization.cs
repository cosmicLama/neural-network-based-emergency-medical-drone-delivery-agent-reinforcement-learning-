using System;
using UnityEngine;

public static class Normalization
{
    public static float Sigmoid(float val, float scale = 1f)
    {
        float scaled = val * scale;
        return scaled / (1f + Mathf.Abs(scaled));
    }

    public static Vector3 Sigmoid(Vector3 v3, float scale = 1f)
    {
        return new Vector3(
            Sigmoid(v3.x, scale),
            Sigmoid(v3.y, scale),
            Sigmoid(v3.z, scale));
    }

    public static float Tanh(float val, float scale = 1f)
    {
        return (float)Math.Tanh(val * scale);
    }

    public static Vector3 Tanh(Vector3 v3, float scale = 1f)
    {
        return new Vector3(
            Tanh(v3.x, scale),
            Tanh(v3.y, scale),
            Tanh(v3.z, scale));
    }

    public static float Log(float val, float e = (float)Math.E)
    {
        return 2f / (1f + Mathf.Pow(e, -val)) - 1f;
    }

    public static Vector3 Log(Vector3 v3, float e = (float)Math.E)
    {
        return new Vector3(
            Log(v3.x, e),
            Log(v3.y, e),
            Log(v3.z, e));
    }

    public static Vector3 Clamp(Vector3 v)
    {
        return new Vector3(
            Mathf.Clamp(v.x, -1f, 1f),
            Mathf.Clamp(v.y, -1f, 1f),
            Mathf.Clamp(v.z, -1f, 1f));
    }
}
