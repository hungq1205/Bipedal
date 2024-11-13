using System;
using System.Collections;
using UnityEngine;
using TMPro;
using Lib.AI;
using Lib.AI.RL;
using System.Collections.Generic;

public class Humon : UnityAgent
{
    const int expCap = 200;
    const int actionNum = 13;
    const float rotNormalizer = 1f / 9f;
    const float angularVelNormalizer = 1f / 36f;
    static readonly Quaternion quarterRot= Quaternion.Euler(0, 0, 90);

    public Rigidbody2D body, l_LowerLeg, r_LowerLeg, l_UpperLeg, r_UpperLeg, l_Foot, r_Foot;
    public TextMeshProUGUI scoreUI;

    public int expRevise = 20;
    public float learningRate = 0.01f;
    public float discountFactor = 0.95f;
    public float explorationRate = 1f, explorationDecay = 0.99f;
    public float minExplorationRate = 0.1f;
    public float actionFreq = 5;
    public float legSpeed = 13f;
    public float lyingThresholdY = -1.8f;
    public float highLegThresholdY = -0.6f;
    public float deactiveMilisec = 100f;
    public float preambleMilisec = 150f;
    [Range(0f, 2f)] public float timeScale = 1f;
    public bool deterministic, inference;
    public string saveFilepath = "model.json";

    bool active = false, preamble = true;
    float elapsed = 0, lyingElapsed, xPosStaticElapsed, highLegElapsed, centerDif;
    float prevX;

    readonly LinkedList<Experience> exps = new();
    readonly LinkedList<Experience> trajectory = new();

    ExplorationWrapper exploration;
    Momentum opt;

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
        Policy = GetDefaultPolicy();
        PolicyOpt = new Reinforce(Policy);
        exploration = new ExplorationWrapper(PolicyOpt, explorationRate, explorationDecay, actionNum, minExplorationRate);
        PolicyOpt = exploration;
        ResetStates();
    }

    void FixedUpdate()
    {
        if (active)
        {
            if (Mathf.Abs(body.position.x - prevX) < 0.12f)
                xPosStaticElapsed += Time.fixedDeltaTime;
            if (GetPos().y < lyingThresholdY)
                lyingElapsed += Time.fixedDeltaTime;
            if (l_UpperLeg.transform.position.y - body.transform.position.y > highLegThresholdY ||
                r_UpperLeg.transform.position.y - body.transform.position.y > highLegThresholdY)
                highLegElapsed += Time.fixedDeltaTime; 
            centerDif = l_UpperLeg.transform.position.x + (r_UpperLeg.transform.position.x - l_LowerLeg.transform.position.x) * 0.25f;
            centerDif -= body.transform.position.x;
            centerDif = Mathf.Abs(centerDif);

            Score = Env.Evaluate(this, new WalkingEnv.WalkingRecord(GetPos(), xPosStaticElapsed, lyingElapsed, highLegElapsed, centerDif));

            if (Score <= -10)
            {
                Conclude(ConcludeType.Terminate);
                Env.ResetStates();
                ResetStates();
            }

            elapsed += Time.fixedDeltaTime;
            if (elapsed * actionFreq < 1)
                return;
            if (trajectory.Last != null)
            {
                if (trajectory.Last.Previous != null)
                    trajectory.Last.Value.rew = Score - trajectory.Last.Previous.Value.rew;
                else
                    trajectory.Last.Value.rew = Score;
            }

            elapsed = 0;
            exploration.minRate = minExplorationRate;
            TakeAction();
            if (!inference)
                PolicyOpt.Step();
            prevX = body.position.x;

            if (Time.timeScale != timeScale)
                Time.timeScale = timeScale;
        }
    }

    public override IPolicy GetDefaultPolicy()
    {
        opt = new Momentum(learningRate: learningRate, weightDecay: 1e-5f);

        DenseNeuralNetworkBuilder builder = new DenseNeuralNetworkBuilder(17);
        builder.NewLayers(
            new ActivationLayer(256, ActivationFunc.Tanh), new Dropout(0.15f),
            new ActivationLayer(256, ActivationFunc.Tanh), new Dropout(0.15f),
            new ActivationLayer(256, ActivationFunc.Tanh), new Dropout(0.15f),
            new ActivationLayer(128, ActivationFunc.Tanh), new Dropout(0.15f),
            new ActivationLayer(actionNum, ActivationFunc.Softmax)
        );

        var policy = new DenseNeuralNetwork(builder, opt);
        policy.BiasAssignForEach((b, dim) => 0f);
        policy.WeightAssignForEach((w, inDim, outDim) =>
        {
            float stddev = Mathf.Sqrt(6f / (inDim + outDim));
            return UnityEngine.Random.Range(-stddev, stddev);
        });

        return policy;
    }

    public override void ResetStates()
    {
        l_Foot.transform.localPosition = new Vector2(0f, -1.713f);
        l_Foot.transform.localRotation = Quaternion.identity;
        l_Foot.angularVelocity = 0;
        l_Foot.velocity = Vector2.zero;

        r_Foot.transform.localPosition = new Vector2(0f, -1.713f);
        r_Foot.transform.localRotation = Quaternion.identity;
        r_Foot.angularVelocity = 0;
        r_Foot.velocity = Vector2.zero;

        l_LowerLeg.transform.localPosition = new Vector2(0f, -0.92f);
        l_LowerLeg.transform.localRotation = Quaternion.identity;
        l_LowerLeg.angularVelocity = 0;
        l_LowerLeg.velocity = Vector2.zero;

        l_UpperLeg.transform.localPosition = new Vector2(0f, 0.184f);
        l_UpperLeg.transform.localRotation = Quaternion.identity;
        l_UpperLeg.angularVelocity = 0;
        l_UpperLeg.velocity = Vector2.zero;

        r_LowerLeg.transform.localPosition = new Vector2(0f, -0.92f);
        r_LowerLeg.transform.localRotation = Quaternion.identity;
        r_LowerLeg.angularVelocity = 0;
        r_LowerLeg.velocity = Vector2.zero;

        r_UpperLeg.transform.localPosition = new Vector2(0f, 0.184f);
        r_UpperLeg.transform.localRotation = Quaternion.identity;
        r_UpperLeg.angularVelocity = 0;
        r_UpperLeg.velocity = Vector2.zero;

        body.transform.localPosition = new Vector2(0, 0.667f);
        body.transform.localRotation = Quaternion.identity;
        body.angularVelocity = 0;
        body.velocity = Vector2.zero;

        gameObject.SetActive(true);
        ConcludedType = ConcludeType.None;
        preamble = true;
        active = false;
        lyingElapsed = 0;
        xPosStaticElapsed = 0;
        highLegElapsed = 0;
        Score = 0;
        StartCoroutine(Preamble(preambleMilisec));
        StartCoroutine(WaitActive(deactiveMilisec));
    }

    public override Vector2 GetPos() => body.position;

    public override void SetPos(Vector2 pos)
    {
        transform.localPosition = pos;
    }

    public float GetPartSignedAngle(Rigidbody2D rb) => rb.transform.rotation.eulerAngles.z * rotNormalizer;

    public float GetPartAngularVelocity(Rigidbody2D rb) => rb.angularVelocity * angularVelNormalizer;

    public void AddSpin(float value, Rigidbody2D rb)
    {
        var trans = rb.transform;
        var up = trans.up;
        rb.AddForceAtPosition(quarterRot * up * (-value), trans.position - up * trans.localScale.y, ForceMode2D.Impulse);
    }

    public void AddBodyForce(float value, bool neg)
    {
        var trans = body.transform;
        var up = neg ? -trans.up : trans.up;
        body.AddForceAtPosition(quarterRot * up * (-value), trans.position + up * trans.localScale.y, ForceMode2D.Impulse);
    }

    IEnumerator WaitActive(float miliSec = 100)
    {
        yield return new WaitForSeconds(miliSec / 1000f);
        active = true;
    }

    IEnumerator Preamble(float miliSec = 250)
    {
        yield return new WaitForSeconds(miliSec / 1000f);
        preamble = false;
    }

    public override void Conclude(ConcludeType type)
    {
        ConcludedType = type;
        gameObject.SetActive(false);
        Score = Env.Evaluate(this, new WalkingEnv.WalkingRecord(GetPos(), xPosStaticElapsed, lyingElapsed, highLegElapsed, centerDif));
        if (trajectory.Last != null)
            trajectory.Last.Value.rew = Score;

        if (!inference)
            UpdatePolicy();
        ClearTrajectory();
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
        opt.learningRate = learningRate;

        var cur = trajectory.Last.Previous;
        string p = trajectory.Last.Value.rew + " ";
        exps.AddLast(trajectory.Last.Value);
        while (cur != null)
        {
            cur.Value.rew += cur.Next.Value.rew * discountFactor;
            p += cur.Value.rew + " ";
            if (exps.Count >= expCap)
                exps.RemoveFirst();
            exps.AddLast(cur.Value);
            cur = cur.Previous;
        }
        Debug.Log(p);

        var exp = trajectory.First;
        while (exp != null)
        {
            Policy.Update(PolicyOpt.ComputeLoss(exp.Value.obs, exp.Value.act, exp.Value.rew));
            exp = exp.Next;
        }

        int l = exps.Count;
        if (expRevise > 0 && l > expRevise * 1.3f)
        {
            var rand = new System.Random();
            var indices = new int[expRevise];
            for (int i = 0; i < expRevise; i++)
                indices[i] = rand.Next(l);
            Array.Sort(indices);

            var p_exp = exps.First;
            var count = 0;
            for (int i = 0; i < l; i++)
            {
                while (i == indices[count])
                {
                    Policy.Update(PolicyOpt.ComputeLoss(p_exp.Value.obs, p_exp.Value.act, p_exp.Value.rew));
                    if (++count >= expRevise)
                    {
                        i = l;
                        break;
                    }
                }
                p_exp = p_exp.Next;
            }
        }
    }

    void ClearTrajectory()
    {
        trajectory.Clear();
    }

    public override void TakeAction()
    {
        float[] obs = new[] {
            GetPartSignedAngle(l_LowerLeg),
            GetPartSignedAngle(l_UpperLeg),
            GetPartSignedAngle(r_LowerLeg),
            GetPartSignedAngle(r_UpperLeg),
            l_LowerLeg.velocity.x,
            l_UpperLeg.velocity.x,
            r_LowerLeg.velocity.x,
            r_UpperLeg.velocity.x,
            l_LowerLeg.velocity.y,
            l_UpperLeg.velocity.y,
            r_LowerLeg.velocity.y,
            r_UpperLeg.velocity.y,
            l_LowerLeg.transform.position.y * 5f,
            r_LowerLeg.transform.position.y * 5f,
            body.transform.position.y * 5f,
            body.velocity.x,
            body.angularVelocity * angularVelNormalizer
        };

        var outputs = inference ? Policy.Infer(obs) : Policy.Forward(obs);
        int action = -1;
        if (preamble)
            action = UnityEngine.Random.Range(0, actionNum);
        else if (deterministic || inference)
        {
            action = 0;
            for (int i = 1; i < outputs.Length; i++)
            {
                if (outputs[action] < outputs[i])
                    action = i;
            }
        }
        else
        {
            action = PolicyOpt.GetAction(obs);
            Debug.Log(action);
        }

        switch (action)
        {
            case 0: AddSpin(-legSpeed, r_LowerLeg); break;
            case 1: AddSpin(-legSpeed, r_UpperLeg); break;
            case 2: AddSpin(legSpeed, r_LowerLeg); break;
            case 3: AddSpin(legSpeed, r_UpperLeg); break;
            case 4: AddSpin(-legSpeed, l_LowerLeg); break;
            case 5: AddSpin(-legSpeed, l_UpperLeg); break;
            case 6: AddSpin(legSpeed, l_LowerLeg); break;
            case 7: AddSpin(legSpeed, l_UpperLeg); break;
            case 8: AddBodyForce(legSpeed * 0.65f, true); break;
            case 9: AddBodyForce(-legSpeed * 0.65f, true); break;
            case 10: AddBodyForce(legSpeed * 0.65f, false); break;
            case 11: AddBodyForce(-legSpeed * 0.65f, false); break;
            case 12: break;
            default: Debug.LogError("Invalid action"); break;
        }

        if (action >= 0 && action < actionNum)
            trajectory.AddLast(new Experience(obs, 0, action));
    }

    class Experience
    {
        public float[] obs;
        public float rew;
        public int act;

        public Experience(float[] obs, float rew, int act)
        {
            this.obs = obs;
            this.rew = rew;
            this.act = act;
        }
    }
}
