using BFLib.AI.RL;
using BFLib.AI;
using System;
using UnityEngine;

public abstract class UnityAgent : MonoBehaviour, IAgent
{
    public event Action<IAgent> onKilled, onActionMade;

    public virtual float score { get; set; }
    public INeuralNetwork policy { get; set; }
    public IEnvironment Env { get; private set; }
    public IPolicy Policy { get; private set; }
    public IPolicyOptimization PolicyOpt { get; private set; }

    public abstract void Kill();

    public abstract void Hide(bool value);

    public abstract Transform GetAgentTransform();

    public abstract Vector2 GetPos();

    public abstract void SetPos(Vector2 pos);

    public abstract INeuralNetwork GetDefaultNeuralNetwork();

    public abstract void ResetStates();

    public abstract (float[], float) TakeAction(int action);

    protected void OnActionMade() => onActionMade?.Invoke(this);

    protected void OnKilled() => onKilled?.Invoke(this);
}
