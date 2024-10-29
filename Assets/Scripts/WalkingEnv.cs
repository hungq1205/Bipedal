using BTLib;
using BTLib.AI.NeuroEvolution;
using JetBrains.Annotations;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Reflection;
using System.Runtime.CompilerServices;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;

public class WalkingEnv : UnityEnvironment
{
    public float startingX { get; private set; } = 0.4082f;
    public float mutationRate = 0.1f, timeoutSec = 10f;
    public bool isDemo { get; private set; }

    public GameObject agentPrefab;
    public Transform spawner;
    public int mutateGroupCount = 20, crossGroupCount = 10, eliteNum = 4, endTopologyExploreGen = 10;

    public TextMeshProUGUI instanceLeftUI, genCountUI, timerUI;
    public TextMeshProUGUI[] highscoreUIs;

    Scoreboard scoreboard;
    UnityAgent demoAgent;
    CameraFollow cam;

    float timer;

    int _genCount;
    int genCount
    {
        get => _genCount;
        set
        {
            _genCount = value;
            genCountUI.text = "gen " + _genCount.ToString();
        }
    }

    int _instanceLeft;
    int instanceLeft
    {
        get => _instanceLeft;
        set
        {
            _instanceLeft = value;
            instanceLeftUI.text = _instanceLeft.ToString();
            if (_instanceLeft == 0)
                NextGen();
        }
    }

    public override void Initialize()
    {
        scoreboard = new Scoreboard(eliteNum);

        unityAgents = new UnityAgent[mutateGroupCount + crossGroupCount + eliteNum];
        for (int i = 0; i < mutateGroupCount + crossGroupCount + eliteNum; i++)
        {
            unityAgents[i] = (UnityAgent)SpawnAgent();
            unityAgents[i].policy.Mutate(mutationRate, numMutator);
        }

        demoAgent = (UnityAgent)SpawnAgent();
        demoAgent.policy.Mutate(mutationRate, numMutator);
        cam = FindObjectOfType<CameraFollow>();
        cam.target = demoAgent.GetAgentTransform();

        instanceLeft = mutateGroupCount + crossGroupCount + eliteNum + 1; // +1 for demo agent
        genCount = 1;
        timer = timeoutSec;
    }

    public override void Evaluate(IAgent agent)
    {
        agent.score = ((UnityAgent)agent).GetPos().x + 20;
    }

    public override void ResetAgent(IAgent agent)
    {
        UnityAgent unboxedAgent;
        if (agent is UnityAgent)
        {
            unboxedAgent = (UnityAgent)agent;

            unboxedAgent.score = 0;
            unboxedAgent.SetPos(new Vector2(startingX, 0.1f));
            unboxedAgent.ResetPhenotype();
            unboxedAgent.gameObject.SetActive(true);
        }
    }

    public override IAgent SpawnAgent()
    {
        UnityAgent agent = GameObject.Instantiate(agentPrefab, spawner).GetComponent<UnityAgent>();
        ResetAgent(agent);
        agent.policy = agent.GetDefaultNeuralNetwork();
        agent.onActionMade += Agent_OnActionMade;
        agent.onKilled += Agent_OnKilled;

        return agent;
    }

    private void Agent_OnKilled(IAgent ag)
    {
        ag.score *= .5f;
        RegisterScoreboard(ag);

        instanceLeft--;
    }

    private void Agent_OnActionMade(IAgent ag)
    {
        Evaluate(ag);
    }

    public override void KillAgentAt(int index)
    {
        unityAgents[index].Kill();
    }

    private void Update()
    {
        timer -= Time.deltaTime;
        timerUI.text = Mathf.Round(timer).ToString();

        if(timer < 0)
        {
            NextGen();
            timer = timeoutSec;
        }

        if (Input.GetKeyDown(KeyCode.Space))
            DemoToggle(!isDemo);
        else if (Input.GetKey(KeyCode.Mouse1))
            cam.transform.position += Vector3.right * 20 * Time.deltaTime;
        else if (Input.GetKey(KeyCode.Mouse0))
            cam.transform.position -= Vector3.right * 20 * Time.deltaTime;
    }

    public void DemoToggle(bool isDemo)
    {
        this.isDemo = isDemo;

        for (int i = 0; i < mutateGroupCount + crossGroupCount + eliteNum; i++)
            unityAgents[i].Hide(isDemo);
    }

    public override void NextGen()
    {
        for(int i = 0; i < mutateGroupCount + crossGroupCount + eliteNum; i++)
            RegisterScoreboard(unityAgents[i]);

        ResetAgent(demoAgent);
        if (scoreboard.ExistsIndex(0))
            demoAgent.policy = scoreboard[0].DeepClone();

        INeuralNetwork randPolicy;

        for (int i = 0; i < mutateGroupCount; i++)
        {
            ResetAgent(unityAgents[i]);

            randPolicy = scoreboard.PickRandomPolicy();
            if (randPolicy != null)
                unityAgents[i].policy = randPolicy.DeepClone();
            unityAgents[i].policy.Mutate(mutationRate, numMutator);
        }

        for (int i = mutateGroupCount; i < mutateGroupCount + crossGroupCount; i++)
        {
            ResetAgent(unityAgents[i]);

            randPolicy = scoreboard.PickRandomPolicy();
            if (randPolicy != null)
            {
                if (Randomizer.singleton.RandomBool())
                    unityAgents[i].policy.Mutate(mutationRate, numMutator);

                unityAgents[i].policy.Crossover(randPolicy);
            }
            else
                unityAgents[i].policy.Mutate(mutationRate, numMutator);
        }

        for (int i = mutateGroupCount + crossGroupCount; i < mutateGroupCount + crossGroupCount + eliteNum; i++)
        {
            ResetAgent(unityAgents[i]);
            unityAgents[i].policy = scoreboard.unstructedPolicies[i - mutateGroupCount - crossGroupCount].DeepClone();
        }

        instanceLeft = mutateGroupCount + crossGroupCount + eliteNum + 1;
        genCount++;

        if(genCount == endTopologyExploreGen)
        {
            for (int i = 0; i < mutateGroupCount + crossGroupCount + eliteNum; i++)
                ((NEAT_NN)unityAgents[i].policy).topologyMutationRateInfo = GlobalVar.defaultRateInfo;
        }

        ResetHighscores();
    }

    void RegisterScoreboard(IAgent agent)
    {
        scoreboard.Register(agent);

        for (int i = 0; i < highscoreUIs.LongLength && scoreboard.ExistsIndex(i); i++)
            highscoreUIs[i].text = string.Format("{0}. {1}", i + 1, scoreboard.GetScoreAtScoreBoardIndex(i));
    }

    void ResetHighscores()
    {
        scoreboard = new Scoreboard(eliteNum);
        for (int i = 0; i < highscoreUIs.LongLength; i++)
            highscoreUIs[i].text = i + ". -";
    }

    class Scoreboard
    {
        public readonly int size;

        LinkedList<ScoreboardEntry> hscores;
        INeuralNetwork[] policies;
        int curPolicyIndex = -1;

        public Scoreboard(int size)
        {
            this.size = size;
            hscores = new LinkedList<ScoreboardEntry>();
            policies = new INeuralNetwork[size];
        }

        public void Register(IAgent agent)
        {
            LinkedListNode<ScoreboardEntry> hscore = hscores.First;
            while (hscore != null && hscore.Value.score >= agent.score)
                hscore = hscore.Next;

            if (hscore == null)
            {
                if (curPolicyIndex < size - 1)
                {
                    policies[++curPolicyIndex] = agent.policy.DeepClone();
                    hscores.AddLast(new ScoreboardEntry(curPolicyIndex, agent.score));
                }
            }
            else
            {
                if (curPolicyIndex == size - 1)
                {
                    policies[curPolicyIndex] = agent.policy.DeepClone();
                    hscore.Value = new ScoreboardEntry(curPolicyIndex, agent.score);
                }
                else
                {
                    policies[++curPolicyIndex] = agent.policy.DeepClone();
                    hscores.AddBefore(hscore, new ScoreboardEntry(curPolicyIndex, agent.score));
                }
            }
        }

        public INeuralNetwork[] unstructedPolicies => policies;

        public bool ExistsIndex(int index) => index <= curPolicyIndex;

        public INeuralNetwork PickRandomPolicy()
        {
            if(curPolicyIndex != -1)
                return policies[Randomizer.singleton.RandomRange(0, curPolicyIndex - 1)];

            return null;
        }

        public float GetScoreAtScoreBoardIndex(int index) => GetEntryAtScoreboardIndex(index).score;

        public INeuralNetwork this[int index] => policies[GetEntryAtScoreboardIndex(index).index];
        
        ScoreboardEntry GetEntryAtScoreboardIndex(int index)
        {
            LinkedListNode<ScoreboardEntry> entry;

            if (index < size / 2)
            {
                entry = hscores.First;
                for (int i = 0; i < index; i++)
                    entry = entry.Next;
            }
            else
            {
                entry = hscores.Last;
                for (int i = size - 1; i > index; i--)
                    entry = entry.Previous;
            }

            return entry.Value;
        }

        struct ScoreboardEntry
        {
            public int index;
            public float score;

            public ScoreboardEntry(int index, float score)
            {
                this.index = index;
                this.score = score;
            }
        }
    }
}
