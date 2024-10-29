using BTLib.AI;
using BTLib.AI.NeuroEvolution;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class UnityEnvironment : MonoBehaviour, IEnvironment
{
    public UnityAgent[] unityAgents { get; protected set; }
    public INumMutator numMutator { get; set; }

    IAgent[] IEnvironment.agents => unityAgents;

    private void Start()
    {
        Initialize();
        numMutator = new GaussianAdditiveNumMutator(.1f);
    }

    public abstract void Initialize();

    public abstract void NextGen();

    public abstract void Evaluate(IAgent agent);

    public abstract void ResetAgent(IAgent agent);

    public abstract IAgent SpawnAgent();

    public abstract void KillAgentAt(int index);
}