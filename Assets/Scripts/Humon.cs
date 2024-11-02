using System;
using System.Collections;
using UnityEngine;
using TMPro;
using BTLib.AI;
using BTLib.AI.RL;
using System.Collections.Generic;

public class Humon : UnityAgent
{
    const float angleNormalizer = 1 / 90f;

    public Rigidbody2D body, l_LowerLeg, r_LowerLeg, l_UpperLeg, r_UpperLeg;
    public TextMeshProUGUI scoreUI;

    public float learningRate = 0.01f;
    public float discountFactor = 0.95f;
    public float explorationRate = 1f, explorationDecay = 0.99f;
    public float actionFreq = 4;
    public float legSpeed = 13f;
    public bool deterministic;

    bool active;
    float elapsed = 0;

    readonly LinkedList<float[]> trajObs = new();
    readonly LinkedList<float> trajRews = new();
    readonly LinkedList<int> trajActs = new();

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
            Conclude(ConcludeType.Killed);
            Env.ResetStates();
            humon.ResetStates();
        };

        active = false;
        learningRate = 0.1f;
        Policy = GetDefaultPolicy();
        PolicyOpt = new Reinforce(Policy);
        //PolicyOpt = new ExplorationWrapper(PolicyOpt, explorationRate, explorationDecay);
        //StartCoroutine(WaitActive());

        int ni = 8, no = 8, ep = 100;
        var x = new float[1][];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = new float[ni];
            for (int j = 0; j < x[i].Length; j++)
                x[i][j] = UnityEngine.Random.Range(0, 10f);
        }

        var y = new float[x.Length][];
        for (int i = 0; i < x.Length; i++)
        {
            y[i] = new float[no];
            for (int j = 0; j < x[i].Length; j++)
                y[i][j] = UnityEngine.Random.Range(0, 2);
        }

        for (int i = 0; i < x.Length; i++)
            Debug.Log("y: " + string.Join(", ", y[i]));
        var prev = Policy.Forward(x[0]);
        Debug.Log("o_pre: " + string.Join(", ", prev));

        for (int e = 0; e < ep; e++)
        {
            for (int i = 0; i < x.Length; i++)
            {
                var o = Policy.Forward(x[i]);
                Debug.Log("o_" + e + ": " + string.Join(", ", o));
                var l = new float[o.Length];
                for (int j = 0; j < o.Length; j++)
                    l[j] = 2 * (o[j] - y[i][j]);
                Policy.Update(l);
            }
        }
        var cur = Policy.Forward(x[0]);
        Debug.Log("o_cur: " + string.Join(", ", cur));
    }

    void FixedUpdate()
    {
        if (active)
        {
            Score = Env.Evaluate(this);
            elapsed += Time.fixedDeltaTime;
            if (elapsed * actionFreq < 1)
                return;
            if (trajActs.First != null)
            {
                if (trajRews.Last != null)
                    trajRews.AddLast(Score - trajRews.Last.Value);
                else
                    trajRews.AddLast(Score);
            }
            elapsed = 0;
            TakeAction();
        }
    }

    public override IPolicy GetDefaultPolicy()
    {
        Optimizer opt = new SGD(learningRate: learningRate);

        DenseNeuralNetworkBuilder builder = new DenseNeuralNetworkBuilder(8);
        builder.NewLayers(
            new ActivationLayer(32, ActivationFunc.ReLU),
            //new ActivationLayer(32, ActivationFunc.ReLU),
            //new ActivationLayer(32, ActivationFunc.ReLU),
            new ActivationLayer(8, ActivationFunc.Softmax)
        );

        var policy = new DenseNeuralNetwork(builder, opt);
        policy.BiasAssignForEach((b, dim) => 0f);
        policy.WeightAssignForEach((w, inDim, outDim) =>
        {
            float stddev = Mathf.Sqrt(0.1f);
            return UnityEngine.Random.Range(-stddev, stddev);
        });

        return policy;
    }

    public override void ResetStates()
    {
        ClearTrajectory();

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
        ConcludedType = ConcludeType.None;
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

    public override void Conclude(ConcludeType type)
    {
        gameObject.SetActive(false);
        Score = Env.Evaluate(this);
        ConcludedType = type;

        UpdatePolicy();
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

    void UpdatePolicy()
    {
        float prevRew = trajRews.Last.Value;
        LinkedListNode<float> rewNode = trajRews.Last.Previous;
        while (rewNode != null)
        {
            rewNode.Value += prevRew * discountFactor;
            prevRew = rewNode.Value;
            rewNode = rewNode.Previous;
        }

        rewNode = trajRews.First;
        var actNode = trajActs.First;
        var obsNode = trajObs.First;
        while (rewNode != null)
        {
            Policy.Update(PolicyOpt.ComputeLoss(obsNode.Value, actNode.Value, rewNode.Value));

            rewNode = rewNode.Next;
            actNode = actNode.Next;
            obsNode = obsNode.Next;
        }
    }

    void ClearTrajectory()
    {
        trajObs.Clear();
        trajActs.Clear();
        trajRews.Clear();
    }

    public override void TakeAction()
    {
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

        var outputs = Policy.Forward(obs);

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
            action = PolicyOpt.GetAction(obs);

        switch (action)
        {
            case 0: AddSpin(legSpeed, l_LowerLeg); break;
            case 1: AddSpin(legSpeed, l_UpperLeg); break;
            case 2: AddSpin(legSpeed, r_LowerLeg); break;
            case 3: AddSpin(legSpeed, r_UpperLeg); break;
            case 4: AddSpin(-legSpeed, l_LowerLeg); break;
            case 5: AddSpin(-legSpeed, l_UpperLeg); break;
            case 6: AddSpin(-legSpeed, r_LowerLeg); break;
            case 7: AddSpin(-legSpeed, r_UpperLeg); break;
            default: Debug.LogError("Invalid action"); break;
        }

        trajObs.AddLast(obs);
        trajActs.AddLast(action);
    }
}
