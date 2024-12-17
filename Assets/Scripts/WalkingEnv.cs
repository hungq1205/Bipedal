using TMPro;
using Lib.AI.RL;
using UnityEngine;
using UnityEngine.SocialPlatforms.Impl;

public class WalkingEnv : UnityEnvironment
{
    public float startingX { get; private set; } = 0.4082f;
    public float timeoutSec = 10f, liveThreshold = 5f;
    public UnityAgent agent;

    public TextMeshProUGUI timerUI;
    public GameObject agentPrefab;
    public Transform spawner, goal;

    CameraFollow cam;
    float timer;

    public override void Init()
    {
        cam = FindObjectOfType<CameraFollow>();
        timer = timeoutSec;
        if (agent == null)
            uAgents.Add(SpawnAgent());
        else
        {
            agent.Env = this;
            uAgents.Add(agent);
        }
    }

    public override float Evaluate(IAgent agent, IEnvironment.Record rec)
    {
        var wrec = (WalkingRecord)rec;

        if (wrec.x >= goal.position.x)
        {
            if (agent.ConcludedType != ConcludeType.None)
                return 20;
            agent.Conclude(ConcludeType.Terminate);
        }

        float score = wrec.x + 10f * 1.75f;
        //if (wrec.centerDif < 1.6f)
        //    wrec.centerDif = 0;
        //if (wrec.xPosStaticElapsed < 3f)
        //    wrec.xPosStaticElapsed = 0;
        score -= wrec.lyingElapsed * 6f;
        if (agent.ConcludedType == ConcludeType.Killed)
        {
            score -= 12f;
        }
        score -= (liveThreshold - wrec.liveTime) * 1f;
        if (score > 0)
            score = Mathf.Pow(score / 3, 1.5f);
        return score;
    }

    public override void ResetAgent(IAgent agent)
    {
        UnityAgent unboxedAgent;
        if (agent is UnityAgent)
        {
            unboxedAgent = (UnityAgent)agent;
            unboxedAgent.SetPos(new Vector2(startingX, 0.1f));
            unboxedAgent.ResetStates();
            unboxedAgent.gameObject.SetActive(true);
        }
    }

    public override void ResetStates()
    {
        timer = timeoutSec;
    }

    public UnityAgent SpawnAgent()
    {
        UnityAgent ag = GameObject.Instantiate(agentPrefab, spawner).GetComponent<UnityAgent>();
        ag.Env = this;
        ResetAgent(agent);

        return agent;
    }

    private void Update()
    {
        timer -= Time.deltaTime;
        timerUI.text = Mathf.Round(timer).ToString();

        if(timer < 0)
        {
            timer = timeoutSec;
            foreach (var agent in uAgents)
            {
                agent.Conclude(ConcludeType.Terminate);
            }
            ResetStates();
        }
    }

    public class WalkingRecord : IEnvironment.Record
    {
        public float x, y;
        public float xPosStaticElapsed;
        public float lyingElapsed;
        public float highLegElapsed;
        public float centerDif;
        public float liveTime;

        public WalkingRecord(Vector2 pos, float xPosStaticElapsed, float lyingElapsed, float highLegElapsed, float centerDif, float liveTime)
        {
            x = pos.x;
            y = pos.y;
            this.xPosStaticElapsed = xPosStaticElapsed;
            this.lyingElapsed = lyingElapsed;
            this.highLegElapsed = highLegElapsed;
            this.centerDif = centerDif;
            this.liveTime = liveTime;
        }
    }
}
