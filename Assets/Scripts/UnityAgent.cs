using BTLib.AI.NeuroEvolution;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class UnityAgent : MonoBehaviour, IAgent
{
    public event Action<IAgent> onKilled, onActionMade;

    public virtual float score { get; set; }

    public INeuralNetwork policy { get; set; }

    public abstract void Kill();

    public abstract void Hide(bool value);

    public abstract void ResetPhenotype();

    public abstract Transform GetAgentTransform();

    public abstract Vector2 GetPos();
    public abstract void SetPos(Vector2 pos);

    public abstract INeuralNetwork GetDefaultNeuralNetwork();

    protected void OnActionMade() => onActionMade?.Invoke(this);

    protected void OnKilled() => onKilled?.Invoke(this);
}
