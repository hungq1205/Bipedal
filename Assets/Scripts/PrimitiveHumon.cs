using BTLib.AI;
using System.Collections;
using UnityEngine;
using BTLib.AI.NeuroEvolution;

public class PrimitiveHumon : UnityAgent
{
    NEAT_NN.NEAT_TopologyMutationRateInfo rateInfo = new NEAT_NN.NEAT_TopologyMutationRateInfo(
        addCon: .2f,
        removeCon: .2f,
        addNodeToCon: .15f
        );

    static INeuralNetwork defaultPolicy;

    public Rigidbody2D body, l_Leg, r_Leg;

    public float legSpeed = 1f;

    bool active;

    //private void Start()
    //{
    //    GetComponentInChildren<HumonBody>().onLyingGround += humon =>
    //    {
    //        Kill();
    //    };
    //}

    protected void Start()
    {
        GetComponentInChildren<HumonBody>().onLyingGround += humon =>
        {
            Kill();
        };

        active = false;
        StartCoroutine(WaitActive(2));
    }

    void FixedUpdate()
    {
        if (active)
        {
            float[] outputs = Policy.Predict(GetLegSignedAngle(true), GetLegSignedAngle(false), body.position.y);

            AddSpin(outputs[0] * legSpeed, true);
            AddSpin(outputs[1] * legSpeed, false);

            OnActionMade();
        }
    }

    public override INeuralNetwork GetDefaultNeuralNetwork()
    {
        if (defaultPolicy == null)
            defaultPolicy = new NEAT_NN(new NEAT_NN.Genotype(3, 2, ActivationFunc.Linear, ActivationFunc.Tanh), rateInfo);

        return defaultPolicy.DeepClone();
    }

    public override void Hide(bool value)
    {
        l_Leg.GetComponent<SpriteRenderer>().enabled = !value;
        r_Leg.GetComponent<SpriteRenderer>().enabled = !value;
        body.GetComponent<SpriteRenderer>().enabled = !value;
    }

    public override void ResetPhenotype()
    {
        l_Leg.transform.localPosition = new Vector2(-0.109999999f, -1.91999996f);
        l_Leg.angularVelocity = 0;
        l_Leg.velocity = Vector2.zero;
        l_Leg.transform.localRotation = Quaternion.identity;

        JointMotor2D l_motor = l_Leg.GetComponent<HingeJoint2D>().motor;
        l_motor.motorSpeed = 0;
        l_Leg.GetComponent<HingeJoint2D>().motor = l_motor;

        r_Leg.transform.localPosition = new Vector2(0.920000017f, -1.91999996f);
        r_Leg.angularVelocity = 0;
        r_Leg.velocity = Vector2.zero;
        r_Leg.transform.localRotation = Quaternion.identity;

        JointMotor2D r_motor = r_Leg.GetComponent<HingeJoint2D>().motor;
        r_motor.motorSpeed = 0;
        r_Leg.GetComponent<HingeJoint2D>().motor = r_motor;

        body.transform.localPosition = new Vector2(0.408199996f, -0.408199996f);
        body.angularVelocity = 0;
        body.velocity = Vector2.zero;
        body.transform.localRotation = Quaternion.identity;
    }

    public override void Kill()
    {
        OnKilled();
        gameObject.SetActive(false);
    }

    public override void SetPos(Vector2 pos)
    {
        transform.localPosition = pos;
    }

    public float GetBodySignedAngle() => Vector2.SignedAngle(Vector2.up, body.transform.up);

    public float GetLegSignedAngle(bool isLeft) => isLeft ? Vector2.SignedAngle(Vector2.up, l_Leg.transform.up) : Vector2.SignedAngle(Vector2.up, r_Leg.transform.up);

    public void AddSpin(float value, bool isLeft)
    {
        if (isLeft)
        {
            JointMotor2D motor = l_Leg.GetComponent<HingeJoint2D>().motor;
            motor.motorSpeed = value * legSpeed;
            l_Leg.GetComponent<HingeJoint2D>().motor = motor;
        }
        else
        {
            JointMotor2D motor = r_Leg.GetComponent<HingeJoint2D>().motor;
            motor.motorSpeed = value * legSpeed;
            r_Leg.GetComponent<HingeJoint2D>().motor = motor;
        }
    }

    IEnumerator WaitActive(float miliSec = 750)
    {
        yield return new WaitForSeconds(miliSec / 1000);
        active = true;
    }

    public override Transform GetAgentTransform()
    {
        return body.transform;
    }

    public override Vector2 GetPos()
    {
        return body.transform.position;
    }
}
