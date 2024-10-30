using BFLib.AI.RL;
using BFLib.AI;
using System;
using UnityEngine;

public abstract class UnityAgent : MonoBehaviour, IAgent
{
    public IEnvironment Env { get; private set; }
    public IPolicy Policy { get; set; }
    public IPolicyOptimization PolicyOpt { get; private set; }
    public bool IsKilled { get; protected set; }

    public abstract void Kill();

    public abstract void Hide(bool value);

    public abstract Transform GetAgentTransform();

    public abstract Vector2 GetPos();

    public abstract void SetPos(Vector2 pos);

    public abstract IPolicy GetDefaultPolicy();

    public abstract void ResetStates();

    public abstract void TakeAction();
}
