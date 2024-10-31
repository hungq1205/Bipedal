using System;
using System.Collections;
using UnityEngine;
using TMPro;
using BTLib.AI;
using BTLib.AI.RL;

public class Humon : UnityAgent
{
    const float angleNormalizer = 1 / 90f;

    public Rigidbody2D body, l_LowerLeg, r_LowerLeg, l_UpperLeg, r_UpperLeg;
    public TextMeshProUGUI scoreUI;

    public float learningRate = 0.01f;
    public float actionFreq = 2;
    public float legSpeed = 13f;
    public bool deterministic;

    bool active;
    float elapsed = 0;
    float[] prevObs;
    int prevAction = -1;

    float _score;
    public float Score
    {
        get => _score;
        set
        {
            _score = value;
            scoreUI.text = Mathf.Round(_score).ToString();
        }
    }

    protected void Start()
    {
        GetComponentInChildren<HumonBody>().onLyingGround += humon =>
        {
            Kill();
        };

        active = false;
        Policy = GetDefaultPolicy();
        PolicyOpt = new PolicyGradient(Policy);
        StartCoroutine(WaitActive());
    }

    void FixedUpdate()
    {
        Score = Env.Evaluate(this);
        if (active)
        {
            elapsed += Time.fixedDeltaTime;
            if (elapsed * actionFreq > 1)
                return;
            elapsed = 0;
            TakeAction();
        }
    }

    public override IPolicy GetDefaultPolicy()
    {
        Optimizer opt = new Adam(learningRate: learningRate);

        DenseNeuralNetworkBuilder builder = new DenseNeuralNetworkBuilder(8);
        builder.NewLayers(
            new ActivationLayer(32, ActivationFunc.ReLU),
            new ActivationLayer(32, ActivationFunc.ReLU),
            new ActivationLayer(32, ActivationFunc.ReLU),
            new ActivationLayer(4, ActivationFunc.Softmax)
        );

        var policy = new DenseNeuralNetwork(builder, opt);
        policy.BiasAssignForEach((b, dim) => 0f);
        policy.WeightAssignForEach((w, inDim, outDim) =>
        {
            float stddev = Mathf.Sqrt(6f / (inDim));
            return UnityEngine.Random.Range(-stddev, stddev);
        });

        return policy;
    }

    public override void ResetStates()
    {
        l_LowerLeg.transform.localPosition = new Vector2(-0.977177024f, -0.946023107f);
        l_LowerLeg.angularVelocity = 0;
        l_LowerLeg.velocity = Vector2.zero;
        l_LowerLeg.transform.localRotation = Quaternion.identity;
        SetHingeMotorSpeed(l_LowerLeg, 0);

        l_UpperLeg.transform.localPosition = new Vector2(-0.977177024f, 0.183976889f);
        l_UpperLeg.angularVelocity = 0;
        l_UpperLeg.velocity = Vector2.zero;
        l_UpperLeg.transform.localRotation = Quaternion.identity;
        SetHingeMotorSpeed(l_UpperLeg, 0);

        r_LowerLeg.transform.localPosition = new Vector2(0.0228230059f, -0.946023107f);
        r_LowerLeg.angularVelocity = 0;
        r_LowerLeg.velocity = Vector2.zero;
        r_LowerLeg.transform.localRotation = Quaternion.identity;
        SetHingeMotorSpeed(r_LowerLeg, 0);

        r_UpperLeg.transform.localPosition = new Vector2(0.0228230059f, 0.183976889f);
        r_UpperLeg.angularVelocity = 0;
        r_UpperLeg.velocity = Vector2.zero;
        r_UpperLeg.transform.localRotation = Quaternion.identity;
        SetHingeMotorSpeed(r_UpperLeg, 0);

        body.transform.localPosition = new Vector2(0, 0.667f);
        body.angularVelocity = 0;
        body.velocity = Vector2.zero;
        body.transform.localRotation = Quaternion.identity;

        gameObject.SetActive(true);
        IsKilled = false;
    }

    void SetHingeMotorSpeed(Rigidbody2D target, float value)
    {
        JointMotor2D l_motor = target.GetComponent<HingeJoint2D>().motor;
        l_motor.motorSpeed = value;
        target.GetComponent<HingeJoint2D>().motor = l_motor;
    }

    public override Vector2 GetPos() => body.position;

    public override void SetPos(Vector2 pos)
    {
        transform.localPosition = pos;
    }

    public float GetPartSignedAngle(Rigidbody2D rb) => Vector2.SignedAngle(Vector2.up, rb.transform.up) * angleNormalizer + 1f;

    public float GetPartAngularVelocity(Rigidbody2D rb) => rb.angularVelocity * angleNormalizer;

    public void AddSpin(float value, Rigidbody2D rb)
    {
        SetHingeMotorSpeed(rb, value);
    }

    IEnumerator WaitActive(float miliSec = 750)
    {
        yield return new WaitForSeconds(750 / 1000);
        active = true;
    }

    public override void Kill()
    {
        IsKilled = true;
        gameObject.SetActive(false);
        Score = Env.Evaluate(this);
    }

    public override void Hide(bool value)
    {
        l_LowerLeg.GetComponent<SpriteRenderer>().enabled = !value;
        l_UpperLeg.GetComponent<SpriteRenderer>().enabled = !value;
        r_LowerLeg.GetComponent<SpriteRenderer>().enabled = !value;
        r_UpperLeg.GetComponent<SpriteRenderer>().enabled = !value;
        body.GetComponent<SpriteRenderer>().enabled = !value;
        scoreUI.gameObject.SetActive(!value);
    }

    public override Transform GetAgentTransform()
    {
        return body.transform;
    }

    public override void TakeAction()
    {
        //if (prevAction != -1)
        //    Policy.Update(PolicyOpt.ComputeLoss(prevObs, prevAction, 1f));

        float[] obs = new[] {
            GetPartSignedAngle(l_LowerLeg),
            GetPartSignedAngle(l_UpperLeg),
            GetPartSignedAngle(r_LowerLeg),
            GetPartSignedAngle(r_UpperLeg),
            GetPartAngularVelocity(l_LowerLeg),
            GetPartAngularVelocity(l_UpperLeg),
            GetPartAngularVelocity(r_LowerLeg),
            GetPartAngularVelocity(r_UpperLeg)
        };
        prevObs ??= new float[obs.Length];
        for (int i = 0; i < obs.Length; i++)
            prevObs[i] = obs[i];

        Debug.Log("----------------------");
        Debug.Log(string.Join(", ", obs));
        var outputs = Policy.Forward(obs);
        Debug.Log(string.Join(", ", outputs));

        int action = -1;
        if (deterministic)
        {
            float max = -1;
            for (int i = 0; i < outputs.Length; i++)
            {
                if (max < outputs[i])
                {
                    max = outputs[i];
                    action = i;
                }
            }
        }
        else
        {
            float rand = UnityEngine.Random.Range(0, 1f);
            for (int i = 0; i < outputs.Length; i++)
            {
                rand -= outputs[i];
                if (rand <= 0)
                {
                    action = i;
                    break;
                }
            }
        }

        switch (action)
        {
            case 0: AddSpin(legSpeed, l_LowerLeg); break;
            case 1: AddSpin(legSpeed, l_UpperLeg); break;
            case 2: AddSpin(legSpeed, r_LowerLeg); break;
            case 3: AddSpin(legSpeed, r_UpperLeg); break;
            default: Debug.LogError("Invalid action"); break;
        }
        prevAction = action;
    }
}
