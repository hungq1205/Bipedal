using Lib.AI.RL;
using UnityEngine;
using System.Collections.Generic;

public abstract class UnityEnvironment : MonoBehaviour, IEnvironment
{
    public readonly List<UnityAgent> uAgents = new();
    public IEnumerable<IAgent> Agents => uAgents;

    private void Start()
    {
        Init();
    }

    public abstract void Init();

    public abstract float Evaluate(IAgent agent, IEnvironment.Record rec);

    public abstract void ResetStates();

    public abstract void ResetAgent(IAgent agent);
}