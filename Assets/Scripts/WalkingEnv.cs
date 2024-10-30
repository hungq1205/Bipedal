using System.Collections.Generic;
using TMPro;
using BFLib.AI.RL;
using UnityEngine;
using System;

public class WalkingEnv : UnityEnvironment
{
    public float startingX { get; private set; } = 0.4082f;
    public float timeoutSec = 10f;

    public TextMeshProUGUI timerUI;
    public GameObject agentPrefab;
    public Transform spawner;

    CameraFollow cam;
    float timer;

    public override void Init()
    {
        cam = FindObjectOfType<CameraFollow>();
        timer = timeoutSec;
    }

    public override float Evaluate(IAgent agent)
    {
        var uAgent = (UnityAgent)agent;
        float score = uAgent.GetPos().x + 20;
        if (uAgent.IsKilled)
            score *= 0.5f;
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

    public UnityAgent SpawnAgent()
    {
        UnityAgent agent = GameObject.Instantiate(agentPrefab, spawner).GetComponent<UnityAgent>();
        ResetAgent(agent);
        agent.Policy = agent.GetDefaultPolicy();

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
                agent.ResetStates();
            ResetStates();
        }
    }
}
