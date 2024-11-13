using TMPro;
using Lib.AI.RL;
using UnityEngine;
using System;
using UnityEngine.SocialPlatforms.Impl;

public class WalkingEnv : UnityEnvironment
{
    public float startingX { get; private set; } = 0.4082f;
    public float timeoutSec = 10f;
    public UnityAgent agent;

    public TextMeshProUGUI timerUI;
    public GameObject agentPrefab;
    public Transform spawner;

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
        float score = (wrec.x + 10) * 2;
        if (wrec.centerDif < 1.6f)
            wrec.centerDif = 0;
        if (wrec.xPosStaticElapsed < 1f)
            wrec.xPosStaticElapsed = 0;
        score -= wrec.lyingElapsed * (wrec.y * wrec.y + 3f) + wrec.highLegElapsed * 9f + wrec.centerDif * 1f;
        if (agent.ConcludedType == ConcludeType.Killed)
        {
            if (score > 20)
                score *= 0.5f;
            else
                score -= 10f;
        }
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
                agent.ResetStates();
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

        public WalkingRecord(Vector2 pos, float xPosStaticElapsed, float lyingElapsed, float highLegElapsed, float centerDif)
        {
            x = pos.x;
            y = pos.y;
            this.xPosStaticElapsed = xPosStaticElapsed;
            this.lyingElapsed = lyingElapsed;
            this.highLegElapsed = highLegElapsed;
            this.centerDif = centerDif;
        }
    }
}
