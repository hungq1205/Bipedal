using BTLib.AI.NeuroEvolution;
using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using Unity.Mathematics;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UIElements;
using UnityEngine.Windows;
using static BTLib.AI.NeuroEvolution.NEAT_NN;
using static UnityEditor.Experimental.AssetDatabaseExperimental.AssetDatabaseCounters;

namespace BTLib
{
    namespace AI
    {
        namespace NeuroEvolution
        {
            public static class GlobalVar
            {
                public const int ActivationFuncCount = 8;

                public static NEAT_TopologyMutationRateInfo defaultRateInfo = new NEAT_TopologyMutationRateInfo(
                    addCon: .05f,
                    removeCon: .05f,
                    addNodeToCon: .05f
                    );


                public static NEAT_TopologyMutationRateInfo topologyExploreRateInfo = new NEAT_TopologyMutationRateInfo(
                    addCon: .2f,
                    removeCon: .05f,
                    addNodeToCon: .15f
                    );

                public static float[] inputs = new float[5]
                {
                1.2f, 3.3f, .4f, 3.6f, 2.4f
                };
            }

            public interface IEnvironment
            {
                INumMutator numMutator { get; }
                IAgent[] agents { get; }

                void Initialize();

                void NextGen();

                void Evaluate(IAgent agent);

                IAgent SpawnAgent();

                void ResetAgent(IAgent agent);

                void KillAgentAt(int index);
            }

            //public interface INEAT_Environment : IEnvironment
            //{
            //    float nodeMutationRate { get; }
            //    float crossOnlyRate { get; }

            //    int initialPopulation { get; }
            //    int eliteNum { get; }

            //    int[][] species { get; }
            //    float[][] highscores { get; }
            //    INeuralNetwork[][] eliteGroup { get; }

            //    int genCount { get; }
            //    int instanceLeft { get; }

            //    public void NextGen()
            //    {
            //        for (int i = 0; i < initialPopulation; i++)
            //            RegisterScoreboard(agents[i]);

            //        //ResetAgent(demoAgent);
            //        //if (highscores[0] > 0)
            //        //    demoAgent.policy = eliteGroup[0].DeepClone();

            //        for(int spe = 0; spe < species.Length; spe++)
            //        {
            //            for(int i = 0; i < species[spe].Length; i++)
            //            {

            //            }
            //        }

            //        for (int i = 0; i < initialPopulation; i++)
            //        {
            //            ResetAgent(agents[i]);

            //            if (highscores[0] > 0)
            //                agents[i].policy = eliteGroup[Randomizer.singleton.RandomRange(0, eliteNum - 1)].DeepClone();
            //            agents[i].policy.Mutate(nodeMutationRate, numMutator);
            //        }

            //        for (int i = mutateGroupCount; i < mutateGroupCount + crossGroupCount; i++)
            //        {
            //            ResetAgent(unityAgents[i]);

            //            if (highscores[0] > 0)
            //            {
            //                int eliteA = Randomizer.singleton.RandomRange(0, eliteNum - 1), eliteB = Randomizer.singleton.RandomRange(0, eliteNum - 1);

            //                while (eliteA == eliteB)
            //                    eliteB = Randomizer.singleton.RandomRange(0, eliteNum - 1);

            //                unityAgents[i].policy = eliteGroup[eliteA].DeepClone();
            //                unityAgents[i].policy.Crossover(eliteGroup[eliteB]);
            //            }
            //            else
            //                unityAgents[i].policy.Mutate(mutationRate, numMutator);
            //        }

            //        instanceLeft = mutateGroupCount + crossGroupCount;
            //        genCount++;
            //    }

            //    public virtual void RegisterScoreboard(IAgent agent)
            //    {
            //        for (int i = highscores.Length - 1; i >= 0; i--)
            //        {
            //            if (agent.score < highscores[i])
            //            {
            //                if (i != highscores.Length - 1)
            //                {
            //                    highscores[i + 1] = agent.score;
            //                    eliteGroup[i + 1] = agent.policy.DeepClone();

            //                    //if (i + 1 < highscoreUIs.Length)
            //                    //    highscoreUIs[i + 1].text = string.Format("{0}. {1}", i + 2, highscores[i + 1]);
            //                }
            //                break;
            //            }
            //            else if (i == 0)
            //            {
            //                highscores[i] = agent.score;
            //                eliteGroup[i] = agent.policy.DeepClone();

            //                //if (i < highscoreUIs.Length)
            //                //    highscoreUIs[i].text = string.Format("{0}. {1}", i + 1, highscores[i]);
            //            }
            //        }
            //    }

            //    public virtual int CalculateSpecieIndex(int curSpeice, NetworkMoprhInfo info)
            //    {
            //        return curSpecie + Mathf.RoundToInt(CalculateSpecieIndex(info) / 2);
            //    }

            //    public virtual float CalculateCompatibilityDifference(NetworkMoprhInfo info)
            //    {
            //        return info.conCountDif + info.nodeCountDif;
            //    }

            //    public struct NetworkMoprhInfo
            //    {
            //        public int nodeCountDif, conCountDif;

            //        public NetworkCrossInfo(int nodeCountDif, int conCountDif)
            //        {
            //            this.nodeCountDif = nodeCountDif;
            //            this.conCountDif = conCountDif;
            //        }
            //    }
            //}

            public interface IAgent
            {
                public event Action<IAgent> onKilled, onActionMade;

                float score { get; set; }

                INeuralNetwork policy { get; set; }

                void Kill();
            }

            #region Neural Network

            public enum ActivationFunc : int
            {
                Linear = 0,
                Inverse = 1,
                Tanh = 2,
                Sigmoid = 3,
                ReLU = 4,
                Squared = 5,
                Absolute = 6,
                UnsignedStep = 7
            }

            public class NEAT_NN : INeuralNetwork
            {
                public readonly InnovationCounter innoCounter;

                public Genotype genotype { get; private set; }

                public NEAT_TopologyMutationRateInfo topologyMutationRateInfo;

                public NEAT_NN(Genotype genotype, NEAT_TopologyMutationRateInfo topologyMutationRateInfo) : base()
                {
                    this.genotype = genotype;
                    this.topologyMutationRateInfo = topologyMutationRateInfo;
                    this.innoCounter = new InnovationCounter();
                }

                public NEAT_NN(NEAT_NN network)
                {
                    genotype = network.genotype.DeepClone();
                    innoCounter = network.innoCounter.DeepClone();
                    topologyMutationRateInfo = network.topologyMutationRateInfo;
                }

                public NEAT_NN(NEAT_NN_Data data, bool deepClone = true)
                {
                    if (deepClone)
                        genotype = data.genotype.DeepClone();
                    else
                        genotype = data.genotype;

                    innoCounter = new InnovationCounter(data.curInnoCount);
                    topologyMutationRateInfo = data.topologyMutationRateInfo;
                }

                public NEAT_NN() { }

                public void Crossover(INeuralNetwork network)
                {
                    if (!(network is NEAT_NN))
                        return;
                    NEAT_NN neatNetwork = (NEAT_NN)network;

                    if (neatNetwork.genotype.inputCount != genotype.inputCount || neatNetwork.genotype.outputCount != genotype.outputCount)
                        return;

                    innoCounter.Increase();

                    InnovationList<ConnectionGene> mateCons = neatNetwork.genotype.connections;

                    foreach (int innovationNum in mateCons.Keys)
                        if (Randomizer.singleton.RandomBool())
                        {
                            genotype.connections[innovationNum] = new List<ConnectionGene>();

                            for (int i = 0; i < mateCons[innovationNum].Count; i++)
                            {
                                ConnectionGene targetCon = mateCons[innovationNum][i];

                                if (!CreateNodeAndMismatchOrigin(targetCon.outputNodeID, neatNetwork) ||
                                    !CreateNodeAndMismatchOrigin(targetCon.inputNodeID, neatNetwork))
                                    continue;

                                if (!genotype.ExistDescendedNode(targetCon.inputNodeID, targetCon.outputNodeID))
                                    genotype.connections.Add(innovationNum, targetCon.DeepClone());
                            }
                        }
                }

                /// <summary>
                /// 
                /// </summary>
                /// <param name="nodeID"></param>
                /// <param name="sourceNet"></param>
                /// <returns>Whether or not the operation creates circular connection chain</returns>
                bool CreateNodeAndMismatchOrigin(int nodeID, NEAT_NN sourceNet)
                {
                    // error-prone: sourceNet passing 
                    NodeGene sourceNode = sourceNet.genotype.FindNode(nodeID);
                    NodeGene fixNode = genotype.FindNode(nodeID);
                    ConnectionGene targetOutCon = sourceNode.FindOutConnection(sourceNode.origin.outNodeID);
                    ConnectionGene targetInCon = sourceNode.FindOutConnection(sourceNode.origin.inNodeID);
                    ConnectionGene outCon, inCon;

                    if (fixNode == null)
                    {
                        genotype.nodes.Add(sourceNode.innovationNum, sourceNode.DeepClone());

                        if (targetOutCon != null)
                        {
                            if (!CreateNodeAndMismatchOrigin(sourceNode.origin.outNodeID, sourceNet))
                                return false;

                            if (genotype.ExistDescendedNode(sourceNode.nodeID, sourceNode.origin.outNodeID))
                                return false;

                            outCon = new ConnectionGene(
                                sourceNode.nodeID,
                                sourceNode.origin.outNodeID,
                                targetOutCon.weight);
                            genotype.connections.Add(targetOutCon.innovationNum, outCon);
                        }

                        if (targetInCon != null)
                        {
                            if (!CreateNodeAndMismatchOrigin(sourceNode.origin.inNodeID, sourceNet))
                                return false;

                            if (genotype.ExistDescendedNode(sourceNode.origin.inNodeID, sourceNode.nodeID))
                                return false;

                            inCon = new ConnectionGene(
                                sourceNode.origin.inNodeID,
                                sourceNode.nodeID,
                                targetInCon.weight);
                            genotype.connections.Add(targetInCon.innovationNum, inCon);
                        }

                        return true;
                    }
                    else if (fixNode.type == NodeType.Input)
                        return true;

                    if (targetInCon != null)
                    {
                        inCon = fixNode.FindInConnection(targetInCon.inputNodeID);

                        if (inCon == null)
                        {
                            if (!CreateNodeAndMismatchOrigin(sourceNode.origin.inNodeID, sourceNet))
                                return false;

                            if (genotype.ExistDescendedNode(sourceNode.origin.inNodeID, nodeID))
                                return false;

                            inCon = new ConnectionGene(
                                sourceNode.origin.inNodeID,
                                nodeID,
                                targetInCon.weight);
                            genotype.connections.Add(targetInCon.innovationNum, inCon);
                        }

                        inCon.weight = targetInCon.weight;
                    }

                    if (targetOutCon != null)
                    {
                        outCon = fixNode.FindOutConnection(targetOutCon.outputNodeID);

                        if (outCon == null)
                        {
                            if(!CreateNodeAndMismatchOrigin(sourceNode.origin.outNodeID, sourceNet))
                                return false;

                            if (genotype.ExistDescendedNode(nodeID, sourceNode.origin.outNodeID))
                                return false;

                            outCon = new ConnectionGene(
                                nodeID,
                                sourceNode.origin.outNodeID,
                                targetOutCon.weight);
                            genotype.connections.Add(targetOutCon.innovationNum, outCon);
                        }

                        outCon.weight = targetOutCon.weight;
                    }

                    fixNode.bias = sourceNode.bias;
                    return true;
                }

                public void Mutate(float mutationRate, INumMutator mutator = null)
                {
                    innoCounter.Increase();

                    MutateBiasesWeights(mutationRate, mutator);
                    MutateTopology(topologyMutationRateInfo);
                }

                void MutateTopology(NEAT_TopologyMutationRateInfo rateInfo)
                {
                    Mutate_RemoveCon(rateInfo.removeCon);
                    Mutate_AddNodeToConnection(rateInfo.addNodeToCon);
                    Mutate_AddCon(rateInfo.addCon);
                }

                void Mutate_AddNodeToConnection(float rate)
                {
                    rate -= Randomizer.singleton.RandomFloat();

                    while (rate > 0)
                    {
                        ConnectionInnoBatch selected = genotype.PickRandomConnection();
                        NodeGene node = RandomNode(new NodeOrigin(selected.connection.inputNodeID, selected.connection.outputNodeID));

                        genotype.connections.RemoveAt(selected.innovationNum, selected.innerListIndex);
                        genotype.nodes.Add(innoCounter.innovationNum, node);

                        if (Randomizer.singleton.RandomBool())
                        {
                            ConnectionGene inCon = new ConnectionGene(selected.connection.inputNodeID, node.nodeID, selected.connection.weight);
                            ConnectionGene outCon = new ConnectionGene(node.nodeID, selected.connection.outputNodeID, GenerateRandWeightBias());

                            genotype.connections.Add(innoCounter.innovationNum, inCon);
                            genotype.connections.Add(innoCounter.innovationNum, outCon);
                        }
                        else
                        {
                            ConnectionGene inCon = new ConnectionGene(selected.connection.inputNodeID, node.nodeID, GenerateRandWeightBias());
                            ConnectionGene outCon = new ConnectionGene(node.nodeID, selected.connection.outputNodeID, selected.connection.weight);

                            genotype.connections.Add(innoCounter.innovationNum, inCon);
                            genotype.connections.Add(innoCounter.innovationNum, outCon);
                        }

                        rate -= Randomizer.singleton.RandomFloat();
                    }
                }

                void Mutate_RemoveCon(float rate)
                {
                    float rand = Randomizer.singleton.RandomFloat();

                    while (rate >= rand)
                    {
                        ConnectionInnoBatch selected;
                        int sampleCount = 0;

                        do
                        {
                            selected = genotype.PickRandomConnection();

                            if (++sampleCount < 15)
                                return;
                        } while (
                            NodeGeneIdentifier.Get(selected.connection.inputNodeID).type != NodeType.Hidden &&
                            NodeGeneIdentifier.Get(selected.connection.outputNodeID).type != NodeType.Hidden);

                        genotype.connections.RemoveAt(selected.innovationNum, selected.innerListIndex);

                        rand += Randomizer.singleton.RandomFloat();
                    }
                }

                void Mutate_AddCon(float rate)
                {
                    rate -= Randomizer.singleton.RandomFloat();

                    while (rate > 0)
                    {
                        NodeGene nodeA = genotype.PickRandomNode(), nodeB;

                        do
                        {
                            if (nodeA.type == NodeType.Output)
                            {
                                nodeB = nodeA;
                                nodeA = genotype.PickRandomNode(true, false);
                            }
                            else if (nodeA.type == NodeType.Input)
                                nodeB = genotype.PickRandomNode(false, true);
                            else
                            {
                                nodeB = genotype.PickRandomNode();

                                if (nodeB.type == NodeType.Input)
                                {
                                    NodeGene temp = nodeA;
                                    nodeA = nodeB;
                                    nodeB = temp;
                                }
                            }
                        }
                        while (genotype.ExistDescendedNode(nodeA.nodeID, nodeB.nodeID));

                        genotype.connections.Add(innoCounter.innovationNum, new ConnectionGene(nodeA.nodeID, nodeB.nodeID, GenerateRandWeightBias()));

                        rate -= Randomizer.singleton.RandomFloat();
                    }
                }

                void MutateBiasesWeights(float mutationRate, INumMutator mutator = null)
                {
                    int biasCount, biasWeightCount = genotype.GetWeightBiasCount(out biasCount);
                    float overallRate = 1 - BTMath.singleton.Pow(1 - mutationRate, biasWeightCount);
                    float randPick = Randomizer.singleton.RandomFloat();

                    while (overallRate > randPick)
                    {
                        if (Randomizer.singleton.RandomFloat() < biasCount / biasWeightCount)
                        {
                            NodeGene selected = genotype.PickRandomNode(false, true);

                            selected.bias =
                                mutator == null ? GenerateRandWeightBias() : mutator.Mutate(selected.bias);

                            if (selected.type == NodeType.Hidden)
                                selected.func = (ActivationFunc)Randomizer.singleton.RandomRange(0, GlobalVar.ActivationFuncCount);
                        }
                        else
                        {
                            ConnectionInnoBatch selected = genotype.PickRandomConnection();

                            selected.connection.weight =
                                mutator == null ? GenerateRandWeightBias() : mutator.Mutate(selected.connection.weight);
                        }

                        randPick += Randomizer.singleton.RandomFloat();
                    }
                }

                public float[] Predict(params float[] inputs)
                {
                    float[] outputs = new float[genotype.outputCount];

                    // iterate output nodes
                    for (int i = genotype.inputCount; i < genotype.inputCount + genotype.outputCount; i++)
                    {
                        outputs[i - genotype.inputCount] = ForwardTo(i, inputs, new NodeListBuffer());
                    }

                    return outputs;
                }

                float ForwardTo(int nodeID, float[] inputs, NodeListBuffer buffer)
                {
                    NodeGene node = genotype.FindNode(nodeID);

                    // inputs into buffer
                    if (node.type == NodeType.Input)
                        return node.Forward(inputs[nodeID]);
                    else if (buffer.CheckIterated(nodeID))
                        return buffer[nodeID];
                    else
                    {
                        float output = node.bias;

                        for (int i = 0; i < node.inCons.Count; i++)
                            output += node.inCons[i].weight * ForwardTo(node.inCons[i].inputNodeID, inputs, buffer);
                        buffer[nodeID] = node.Forward(output);

                        return buffer[nodeID];
                    }
                }

                public string ToJson()
                {
                    return JsonUtility.ToJson(new NEAT_NN_Data(this));
                }

                public T FromJson<T>(string json)
                {
                    return (T)JsonUtility.FromJson(json, typeof(T));
                }

                [Serializable]
                public class NEAT_NN_Data
                {
                    public int curInnoCount;
                    public Genotype genotype;
                    public NEAT_TopologyMutationRateInfo topologyMutationRateInfo;

                    public NEAT_NN_Data(NEAT_NN neatNetwork)
                    {
                        this.curInnoCount = neatNetwork.innoCounter.innovationNum;
                        this.genotype = neatNetwork.genotype;
                        this.topologyMutationRateInfo = neatNetwork.topologyMutationRateInfo;
                    }
                }

                public INeuralNetwork DeepClone()
                {
                    return new NEAT_NN(this);
                }

                public NodeGene RandomNode(NodeOrigin origin)
                {
                    return new NodeGene(NodeType.Hidden, GenerateRandWeightBias(), origin, (ActivationFunc)Randomizer.singleton.RandomRange(0, GlobalVar.ActivationFuncCount - 1));
                }

                public NodeGene RandomNode()
                {
                    return new NodeGene(NodeType.Hidden, GenerateRandWeightBias(), (ActivationFunc)Randomizer.singleton.RandomRange(0, GlobalVar.ActivationFuncCount - 1));
                }

                public static float GenerateRandWeightBias() => Randomizer.singleton.RandomFloat() * 2 - 1;

                [Serializable]
                public struct NEAT_TopologyMutationRateInfo
                {
                    public float addCon, removeCon, addNodeToCon;

                    public NEAT_TopologyMutationRateInfo(float addCon, float removeCon, float addNodeToCon)
                    {
                        this.addCon = addCon;
                        this.removeCon = removeCon;
                        this.addNodeToCon = addNodeToCon;
                    }
                }

                public class NodeListBuffer
                {
                    Dictionary<int, float> outputs;

                    public NodeListBuffer()
                    {
                        outputs = new Dictionary<int, float>();
                    }

                    /// <summary>
                    /// Assign value of non-NaN value will also assign correlated <b>isIterated</b> to true, otherwise false
                    /// </summary>
                    /// <returns>Saved <b>output</b> of the index if <b>isIterated</b>, otherwise NaN</returns>
                    public float this[int id]
                    {
                        get => outputs[id];
                        set
                        {
                            if (outputs.ContainsKey(id))
                                outputs[id] = value;
                            else
                                outputs.Add(id, value);
                        }
                    }

                    public void SetIterated(int id, bool value)
                    {
                        if(!value)
                            outputs[id] = float.NaN;
                    }

                    public bool CheckIterated(int id)
                    {
                        if (outputs.ContainsKey(id))
                            return outputs[id] != float.NaN;
                        else
                            return false;
                    }
                }

                public class InnovationCounter : IDeepClonable<InnovationCounter>
                {
                    int _innovationNum = 0;
                    public int innovationNum => _innovationNum;

                    public InnovationCounter(int start = 0)
                    {
                        _innovationNum = start;
                    }

                    public void Increase() => _innovationNum++;

                    public InnovationCounter DeepClone()
                    {
                        InnovationCounter clone = new InnovationCounter();
                        clone._innovationNum = innovationNum;

                        return clone;
                    }
                }

                [Serializable]
                public class Genotype : IDeepClonable<Genotype>
                {
                    [NonSerialized] public readonly int inputCount;
                    [NonSerialized] public readonly int outputCount;

                    public InnovationList<NodeGene> nodes;
                    public InnovationList<ConnectionGene> connections;

                    public Genotype(int input, int output, ActivationFunc inputFunc = default, ActivationFunc outputFunc = default, bool generateCons = true)
                    {
                        nodes = new InnovationList<NodeGene>();
                        connections = new InnovationList<ConnectionGene>();

                        connections.onAdd += Connection_OnAdd;
                        connections.onRemove += Connections_OnRemove;

                        nodes.onRemove += Nodes_OnRemove;

                        inputCount = input;
                        outputCount = output;

                        for (int i = 0; i < input; i++)
                            nodes.Add(0, new NodeGene(NodeType.Input, GenerateRandWeightBias() * .8f, new NodeOrigin(i), inputFunc));

                        for (int i = 0; i < output; i++)
                        {
                            nodes.Add(0, new NodeGene(NodeType.Output, GenerateRandWeightBias() * .8f, new NodeOrigin(input + i), outputFunc));

                            if (generateCons)
                                for (int j = 0; j < input; j++)
                                    connections.Add(0, new ConnectionGene(j, input + i, GenerateRandWeightBias() * .8f));
                        }
                    }

                    private Genotype(Genotype source)
                    {
                        nodes = new InnovationList<NodeGene>();
                        connections = new InnovationList<ConnectionGene>();

                        connections.onAdd += Connection_OnAdd;
                        connections.onRemove += Connections_OnRemove;

                        nodes.onRemove += Nodes_OnRemove;

                        inputCount = source.inputCount;
                        outputCount = source.outputCount;

                        foreach (int innoNum in source.nodes.Keys)
                        {
                            nodes.Add(innoNum, new List<NodeGene>());

                            for (int i = 0, count = source.nodes[innoNum].Count; i < count; i++)
                                nodes.Add(innoNum, source.nodes[innoNum][i].DeepClone());
                        }

                        foreach (int innoNum in source.connections.Keys)
                        {
                            connections.Add(innoNum, new List<ConnectionGene>());

                            for (int i = 0, count = source.connections[innoNum].Count; i < count; i++)
                                connections.Add(innoNum, source.connections[innoNum][i].DeepClone());
                        }
                    }

                    public int GetWeightBiasCount()
                    {
                        int count = 0;

                        foreach (int innovation in nodes.Keys)
                            count += nodes[innovation].Count;

                        foreach (int innovation in connections.Keys)
                            count += connections[innovation].Count;

                        return count;
                    }

                    public int GetWeightBiasCount(out int biasCount)
                    {
                        int count = 0;

                        foreach (int innovation in nodes.Keys)
                            count += nodes[innovation].Count;

                        biasCount = count;

                        foreach (int innovation in connections.Keys)
                            count += connections[innovation].Count;

                        return count;
                    }

                    public int GetBiasCount()
                    {
                        int count = 0;

                        foreach (int innovation in nodes.Keys)
                            count += nodes[innovation].Count;

                        return count;
                    }

                    public ConnectionInnoBatch PickRandomConnection()
                    {
                        int selectedInno, selectedInnerIndex;

                        return new ConnectionInnoBatch(connections.PickRandom(out selectedInno, out selectedInnerIndex), selectedInno, selectedInnerIndex);
                    }

                    public NodeGene PickRandomNode()
                    {
                        return nodes.PickRandom();
                    }

                    public NodeGene PickRandomNode(bool includeInputs, bool includeOutputs)
                    {
                        int biasCount = GetBiasCount();

                        if (includeInputs)
                        {
                            if (includeOutputs)
                                return PickRandomNode();
                            else if (Randomizer.singleton.RandomFloat() > inputCount / biasCount)
                                return nodes[0][Randomizer.singleton.RandomRange(0, inputCount - 1)];
                            else
                                return nodes.PickRandom(1);
                        }
                        else if (includeOutputs)
                        {
                            if (Randomizer.singleton.RandomFloat() > outputCount / biasCount)
                                return nodes[0][Randomizer.singleton.RandomRange(inputCount, outputCount - 1)];
                            else
                                return nodes.PickRandom(1);
                        }

                        return nodes.PickRandom(1);
                    }

                    public bool ExistDescendedNodeBranch(int fromNodeID, int nodeFindID, List<int> iteratedList = null)
                    {
                        if (iteratedList == null)
                            iteratedList = new List<int>();

                        NodeGene node = FindNode(fromNodeID);

                        for (int i = 0; i < node.inCons.Count; i++)
                            if (ExistDescendedNode(node.inCons[i].inputNodeID, nodeFindID, iteratedList))
                                return true;

                        return false;
                    }

                    public bool ExistDescendedNode(int fromNodeID, int nodeFindID, List<int> iteratedList = null)
                    {
                        if (fromNodeID == nodeFindID)
                            return true;

                        if (iteratedList == null)
                            iteratedList = new List<int>();

                        NodeGene node = FindNode(fromNodeID);

                        if (node.inCons.Count == 0)
                            return false;

                        for (int i = 0, count = iteratedList.Count; i < count; i++)
                            if (fromNodeID == iteratedList[i])
                                return false;

                        for (int i = 0, count = node.inCons.Count; i < count; i++)
                            if (ExistDescendedNode(node.inCons[i].inputNodeID, nodeFindID, iteratedList))
                                return true;

                        iteratedList.Add(fromNodeID);

                        return false;
                    }

                    public struct CircularIteratedInNodeInfo
                    {
                        int nodeID;
                        bool existInNode;

                        public CircularIteratedInNodeInfo(int nodeID, bool existInNode)
                        {
                            this.nodeID = nodeID;
                            this.existInNode = existInNode;
                        }
                    }

                    public Genotype DeepClone()
                    {
                        return new Genotype(this);
                    }

                    public NodeGene FindNode(int nodeID)
                    {
                        if (NodeGeneIdentifier.Get(nodeID) == null)
                            return null;

                        int innovationNum = NodeGeneIdentifier.Get(nodeID).innovationNum;

                        if (!nodes.ContainsKey(innovationNum))
                            return null;

                        for (int i = 0, count = nodes[innovationNum].Count; i < count; i++)
                            if (nodes[innovationNum][i].nodeID == nodeID)
                                return nodes[innovationNum][i];

                        return null;
                    }

                    void Connection_OnAdd(int innovation, ConnectionGene gene)
                    {
                        NodeGene inNode = FindNode(gene.inputNodeID), outNode = FindNode(gene.outputNodeID);

                        if (inNode == null || outNode == null) return;

                        inNode.outCons.Add(gene);
                        outNode.inCons.Add(gene);
                    }

                    // ?: remove tail and head cons as well?
                    private void Connections_OnRemove(int innovation, ConnectionGene gene)
                    {
                        if (connections[innovation].Count == 0)
                            connections.Remove(innovation);

                        FindNode(gene.inputNodeID).outCons.Remove(gene);
                        FindNode(gene.outputNodeID).inCons.Remove(gene);
                    }

                    private void Nodes_OnRemove(int innovation, NodeGene gene)
                    {
                        if (nodes[innovation].Count == 0)
                            nodes.Remove(innovation);
                    }
                }

                public enum NodeType
                {
                    Input,
                    Output,
                    Hidden
                }

                public class NodeGeneIdentifier
                {
                    static List<NodeGeneIdentifier> identifiers = new List<NodeGeneIdentifier>();
                    static List<int> inputOutputIdIndices = new List<int>();
                    static int idCounter = 0;

                    public readonly int id;
                    public readonly int innovationNum;
                    public readonly NodeType type;
                    public readonly NodeOrigin origin;

                    private NodeGeneIdentifier(int innovationNum, NodeType type, NodeOrigin origin)
                    {
                        this.id = idCounter++;
                        this.innovationNum = innovationNum;
                        this.type = type;
                        this.origin = origin;
                    }

                    public static int RegisterNodeID(int innovationNum, NodeType type, NodeOrigin origin)
                    {
                        NodeGeneIdentifier nodeID;

                        if (type != NodeType.Hidden)
                        {
                            int index = origin.GetInOutNodeIndex();

                            if (index != -1 && index < inputOutputIdIndices.Count) 
                            {
                                if (inputOutputIdIndices[index] != -1)
                                    return inputOutputIdIndices[index];
                            }
                            else
                            {
                                do
                                {
                                    inputOutputIdIndices.Add(-1);
                                } while (index >= inputOutputIdIndices.Count);
                            }

                            nodeID = new NodeGeneIdentifier(innovationNum, type, origin);

                            identifiers.Add(nodeID);
                            inputOutputIdIndices[index] = nodeID.id;

                            return nodeID.id;
                        }


                        for (int i = 0, count = identifiers.Count; i < count; i++)
                            if (identifiers[i].innovationNum == innovationNum)
                                if (identifiers[i].origin.Equals(origin))
                                    return i;

                        nodeID = new NodeGeneIdentifier(innovationNum, type, origin);

                        identifiers.Add(nodeID);

                        return nodeID.id;
                    }

                    public static NodeGeneIdentifier Get(int id)
                    {
                        if (id >= 0 && id < identifiers.Count)
                            return identifiers[id];

                        return null;

                    }
                }

                [Serializable]
                public class NodeGene : IDeepClonable<NodeGene>, InnovationListItem
                {
                    public int nodeID { get; private set; } = -1;
                    public ActivationFunc func = ActivationFunc.Linear;
                    public float bias;

                    public NodeType type => NodeGeneIdentifier.Get(nodeID).type;
                    public NodeOrigin origin => NodeGeneIdentifier.Get(nodeID).origin;
                    public int innovationNum => NodeGeneIdentifier.Get(nodeID).innovationNum;

                    [NonSerialized] public List<ConnectionGene> inCons;
                    [NonSerialized] public List<ConnectionGene> outCons;

                    [NonSerialized] NodeGeneDraft draft;

                    public NodeGene(NodeGeneIdentifier nodeID, float bias, ActivationFunc func = ActivationFunc.Linear)
                    {
                        this.nodeID = nodeID.id;
                        this.bias = bias;
                        this.func = func;

                        inCons = new List<ConnectionGene>();
                        outCons = new List<ConnectionGene>();
                    }

                    public NodeGene(NodeType type, float bias, ActivationFunc func = ActivationFunc.Linear)
                    {
                        this.draft = new NodeGeneDraft(type, new NodeOrigin());
                        this.bias = bias;
                        this.func = func;

                        inCons = new List<ConnectionGene>();
                        outCons = new List<ConnectionGene>();
                    }

                    public NodeGene(NodeType type, float bias, NodeOrigin origin, ActivationFunc func = ActivationFunc.Linear)
                    {
                        this.draft = new NodeGeneDraft(type, origin);
                        this.bias = bias;
                        this.func = func;

                        inCons = new List<ConnectionGene>();
                        outCons = new List<ConnectionGene>();
                    }

                    public ConnectionGene FindInConnection(int inNodeID)
                    {
                        for (int i = 0, count = outCons.Count; i < count; i++)
                            if (outCons[i].inputNodeID == inNodeID)
                                return outCons[i];

                        return null;
                    }

                    public ConnectionGene FindOutConnection(int outNodeID)
                    {
                        for (int i = 0, count = inCons.Count; i < count; i++)
                            if (inCons[i].outputNodeID == outNodeID)
                                return inCons[i];

                        return null;
                    }

                    public NodeGene DeepClone()
                    {
                        return new NodeGene(NodeGeneIdentifier.Get(nodeID), bias, func);
                    }

                    public float Forward(float input)
                    {
                        switch (func)
                        {
                            case ActivationFunc.Linear:
                                return input;
                            case ActivationFunc.Tanh:
                                return (2 / (1 + BTMath.singleton.Exp(-2 * input))) - 1;
                            case ActivationFunc.Sigmoid:
                                return 1 / (1 + BTMath.singleton.Exp(-input));
                            case ActivationFunc.Squared:
                                return input * input;
                            case ActivationFunc.Absolute:
                                return input < 0 ? -input : input;
                            case ActivationFunc.ReLU:
                                return input < 0 ? 0 : input;
                            case ActivationFunc.Inverse:
                                return -input;
                            default:
                                return input;
                        }
                    }

                    public void OnAdd(int innovationNum)
                    {
                        if (nodeID == -1)
                            nodeID = NodeGeneIdentifier.RegisterNodeID(innovationNum, draft.type, draft.origin);
                    }

                    struct NodeGeneDraft
                    {
                        public NodeType type;
                        public NodeOrigin origin;

                        public NodeGeneDraft(NodeType type, NodeOrigin origin)
                        {
                            this.type = type;
                            this.origin = origin;
                        }
                    }
                }

                [Serializable]
                public class ConnectionGene : IDeepClonable<ConnectionGene>, InnovationListItem
                {
                    public int inputNodeID, outputNodeID;
                    public float weight;
                    public int innovationNum { get; private set; }

                    public ConnectionGene(int inputNodeID, int outputNodeID, float weight)
                    {
                        this.inputNodeID = inputNodeID;
                        this.outputNodeID = outputNodeID;
                        this.weight = weight;
                    }

                    public ConnectionGene(ConnectionGene connection)
                    {
                        inputNodeID = connection.inputNodeID;
                        outputNodeID = connection.outputNodeID;
                        weight = connection.weight;
                        innovationNum = connection.innovationNum;
                    }

                    public ConnectionGene DeepClone()
                    {
                        return new ConnectionGene(this);
                    }

                    internal void OnAdd(int innovationNum)
                    {
                        this.innovationNum = innovationNum;
                    }
                }

                public interface InnovationListItem
                {
                    virtual void OnAdd(int innovationNum) { }

                    virtual void OnRemove(int innovationNum) { }
                }

                public struct ConnectionInnoBatch
                {
                    public ConnectionGene connection;
                    public int innovationNum, innerListIndex;

                    public ConnectionInnoBatch(ConnectionGene connection, int innovationNum, int innerListIndex)
                    {
                        this.connection = connection;
                        this.innovationNum = innovationNum;
                        this.innerListIndex = innerListIndex;
                    }
                }

                [Serializable]
                public struct NodeOrigin : IEquatable<NodeOrigin>
                {
                    public int inNodeID, outNodeID;

                    public NodeOrigin(int inNodeID, int outNodeID)
                    {
                        this.inNodeID = inNodeID;
                        this.outNodeID = outNodeID;
                    }

                    public NodeOrigin(int inoutNodeIndex)
                    {
                        this.inNodeID = -inoutNodeIndex - 1;
                        this.outNodeID = -inoutNodeIndex - 1;
                    }

                    public int GetInOutNodeIndex() => (inNodeID == outNodeID && inNodeID < 0) ? -inNodeID : -1;

                    public bool Equals(NodeOrigin other) => other.inNodeID == inNodeID && other.outNodeID == outNodeID;
                }
            }

            public class Artificial_NN : INeuralNetwork
            {
                public readonly int biasWeightCount, inputNum, outputNum;

                public NeuralLayer[] biasLayers { get; protected set; }
                public float[][,] weights { get; protected set; }

                public Artificial_NN(params NeuralLayer[] layers) : base()
                {
                    if (layers.LongLength < 2)
                        throw new System.Exception("Neuron network init: insufficient layer count");

                    inputNum = layers[0].Length;
                    outputNum = layers[layers.LongLength - 1].Length;

                    biasLayers = new NeuralLayer[layers.LongLength];
                    for (int i = 0; i < biasLayers.LongLength; i++)
                        biasLayers[i] = (NeuralLayer)layers[i].DeepClone();
                    weights = new float[layers.LongLength - 1][,];

                    // Init hidden layer matrices (layers in-between input & output layers)
                    for (int i = 1; i < layers.LongLength; i++)
                        biasWeightCount += layers[i].Length;

                    // Init weight matrices
                    for (int i = 1; i < layers.LongLength; i++)
                    {
                        biasWeightCount += layers[i - 1].Length * layers[i].Length;
                        weights[i - 1] = new float[layers[i - 1].Length, layers[i].LongLength];
                    }
                }

                public Artificial_NN(float randomizeRange, params NeuralLayer[] layers) : base()
                {
                    if (layers.LongLength < 2)
                        throw new System.Exception("Neuron network init: insufficient layer count");

                    inputNum = layers[0].Length;
                    outputNum = layers[layers.LongLength - 1].Length;

                    biasLayers = new NeuralLayer[layers.LongLength];
                    for (int i = 0; i < biasLayers.LongLength; i++)
                        biasLayers[i] = (NeuralLayer)layers[i].DeepClone();
                    weights = new float[layers.LongLength - 1][,];

                    // Init hidden layer matrices (layers in-between input & output layers)
                    for (int i = 1; i < layers.LongLength; i++)
                    {
                        biasWeightCount += layers[i].Length;

                        for (int j = 0; j < biasLayers[i - 1].LongLength; j++)
                            biasLayers[i - 1].perceptrons[j] = Randomizer.singleton.RandomRange(-randomizeRange, randomizeRange);
                    }

                    // Init weight matrices
                    for (int i = 1; i < layers.LongLength; i++)
                    {
                        biasWeightCount += layers[i - 1].Length * layers[i].Length;
                        weights[i - 1] = new float[layers[i - 1].Length, layers[i].LongLength];

                        for (int j = 0; j < weights[i - 1].GetLongLength(0); j++)
                            for (int k = 0; k < weights[i - 1].GetLongLength(1); k++)
                                weights[i - 1][j, k] = Randomizer.singleton.RandomRange(-randomizeRange, randomizeRange);
                    }
                }

                public void Crossover(INeuralNetwork network)
                {
                    Artificial_NN unboxedNetwork;

                    if (network is Artificial_NN)
                        unboxedNetwork = network as Artificial_NN;
                    else
                        throw new Exception("Can not perform crossover between 2 different type of Neural Network!");

                    if (biasLayers.LongLength != unboxedNetwork.biasLayers.LongLength)
                        throw new Exception("Can not perform crossover between 2 different structures of Artificial Neural Network!");

                    for (int i = 0; i < biasLayers.LongLength; i++)
                        if (biasLayers[i].LongLength != unboxedNetwork.biasLayers[i].LongLength)
                            throw new Exception("Can not perform crossover between 2 different structures of Artificial Neural Network!");

                    for (int layerIndex = 0; layerIndex < biasLayers.LongLength; layerIndex++)
                        for (int biasIndex = 0; biasIndex < biasLayers[layerIndex].LongLength; biasIndex++)
                            if (Randomizer.singleton.RandomBool())
                            {
                                biasLayers[layerIndex].perceptrons[biasIndex] = unboxedNetwork.biasLayers[layerIndex].perceptrons[biasIndex];

                                for (int preLayerBias = 0; preLayerBias < weights[layerIndex].GetLongLength(0); preLayerBias++)
                                    weights[layerIndex][preLayerBias, biasIndex] = unboxedNetwork.weights[layerIndex][preLayerBias, biasIndex];
                            }
                }

                public void Mutate(float mutationRate, INumMutator mutator = null)
                {
                    float overallRate = 1 - BTMath.singleton.Pow(1 - mutationRate, biasWeightCount);
                    float randPick = Randomizer.singleton.RandomFloat();

                    while (overallRate > randPick)
                    {
                        int totalIndices = (int)Math.Round(randPick * (biasWeightCount - 1));
                        int i = 0;

                        do
                        {
                            totalIndices -= biasLayers[i].Length - 1;
                            i++;
                        } while (totalIndices > 0 && i < biasLayers.LongLength);

                        if (totalIndices <= 0)
                        {
                            totalIndices += biasLayers[i - 1].Length - 1;
                            biasLayers[i - 1].perceptrons[totalIndices] = mutator.Mutate(biasLayers[i - 1].perceptrons[totalIndices]);
                        }
                        else
                        {
                            i = 0;

                            do
                            {
                                totalIndices -= weights[i].GetLength(0) + weights[i].GetLength(1) - 2;
                                i++;
                            } while (totalIndices > 0 && i < weights.LongLength);

                            if (totalIndices <= 0)
                            {
                                totalIndices += weights[i - 1].GetLength(0) + weights[i - 1].GetLength(1) - 2;
                                int j = 0;

                                do
                                {
                                    totalIndices -= weights[i - 1].GetLength(1) - 1;
                                    j++;
                                } while (totalIndices > 0 && j < weights.GetLongLength(0));

                                if (totalIndices <= 0)
                                {
                                    totalIndices += weights[i - 1].GetLength(1) - 1;
                                    weights[i - 1][j - 1, totalIndices] += mutator.Mutate(weights[i - 1][j - 1, totalIndices]);
                                }
                            }
                        }

                        randPick += Randomizer.singleton.RandomFloat();
                    }
                }

                public void Randomize(float randomizeRange)
                {
                    for (int i = 0; i < biasLayers.LongLength; i++)
                    {
                        for (int j = 0; j < biasLayers[i].LongLength; j++)
                            biasLayers[i].perceptrons[j] = Randomizer.singleton.RandomRange(-randomizeRange, randomizeRange);
                    }

                    // Init weight matrices
                    for (int i = 0; i < weights.LongLength; i++)
                        for (int j = 0; j < weights[i].GetLongLength(0); j++)
                            for (int k = 0; k < weights[i].GetLongLength(1); k++)
                                weights[i][j, k] = Randomizer.singleton.RandomRange(-randomizeRange, randomizeRange);
                }

                public INeuralNetwork DeepClone()
                {
                    Artificial_NN clone = new Artificial_NN(biasLayers);

                    for (int i = 0; i < clone.biasLayers.LongLength; i++)
                        clone.biasLayers[i] = (NeuralLayer)biasLayers[i].DeepClone();

                    for (int i = 0; i < clone.weights.LongLength; i++)
                        for (int j = 0; j < clone.weights[i].GetLongLength(0); j++)
                            for (int k = 0; k < clone.weights[i].GetLongLength(1); k++)
                                clone.weights[i][j, k] = weights[i][j, k];

                    return clone;
                }

                public string ToJson()
                {
                    throw new NotImplementedException();
                }

                public T FromJson<T>(string json)
                {
                    throw new NotImplementedException();
                }

                #region Forward Propagation
                public float[] Predict(params float[] inputs)
                {
                    if (inputs.LongLength != inputNum)
                        throw new System.Exception("Neuron network fw propagation: mismatched input count");

                    return ForwardPropagateUncheck(inputs);
                }

                float[] ForwardPropagateUncheck(float[] inputs, int startingLayer = 1)
                {
                    if (startingLayer < biasLayers.LongLength)
                        return ForwardPropagateUncheck(SingalForwardPropagateUncheck(startingLayer, inputs), startingLayer + 1);

                    return inputs;
                }

                /// <summary>
                /// Singal propagate from a given layer 
                /// </summary>
                /// <param name="fromLayer">Index starting from 1</param>
                /// <param name="inputs">Inputs of the given layer</param>
                /// <returns></returns>
                /// <exception cref="System.Exception"></exception>
                public float[] SingalForwardProgagate(int fromLayer, float[] inputs)
                {
                    if (fromLayer < 0 || fromLayer >= biasLayers.LongLength + 1)
                        throw new System.Exception("Neuron network singal fw propagation: passed layer is out of bound");

                    if ((fromLayer == 0 && inputs.LongLength != inputNum) ||
                        inputs.LongLength != biasLayers[fromLayer - 1].LongLength)
                        throw new System.Exception("Neuron network singal fw propagation: mismatched input count");

                    return SingalForwardPropagateUncheck(fromLayer, inputs);
                }

                float[] SingalForwardPropagateUncheck(int fromLayer, float[] inputs)
                {
                    float[] outputs = new float[biasLayers[fromLayer].LongLength];

                    for (int i = 0; i < outputs.LongLength; i++)
                    {
                        outputs[i] = biasLayers[fromLayer].perceptrons[i];

                        for (int j = 0; j < inputs.LongLength; j++)
                            outputs[i] += weights[fromLayer - 1][j, i] * inputs[j];

                        outputs[i] = biasLayers[fromLayer].Forward(outputs[i]);
                    }

                    return outputs;
                }

                #endregion

                #region NeuralLayer
                public class NeuralLayer : IDeepClonable<NeuralLayer>
                {
                    public float[] perceptrons;

                    public long LongLength { get => perceptrons.LongLength; }

                    public int Length { get => perceptrons.Length; }

                    public NeuralLayer(int perceptronCount)
                    {
                        perceptrons = new float[perceptronCount];
                    }

                    public virtual NeuralLayer DeepClone()
                    {
                        NeuralLayer clone = new NeuralLayer(Length);

                        for (int i = 0; i < clone.perceptrons.LongLength; i++)
                            clone.perceptrons[i] = perceptrons[i];

                        return clone;
                    }

                    public virtual float Forward(float input)
                    {
                        return input;
                    }

                    public static implicit operator NeuralLayer(int perceptronCount)
                    {
                        return new NeuralLayer(perceptronCount);
                    }
                }

                public class ActivationLayer : NeuralLayer
                {
                    public ActivationFunc func;

                    public ActivationLayer(int perceptron, ActivationFunc func) : base(perceptron)
                    {
                        this.func = func;
                    }

                    public override NeuralLayer DeepClone()
                    {
                        ActivationLayer clone = new ActivationLayer(perceptrons.Length, func);

                        for (int i = 0; i < clone.perceptrons.LongLength; i++)
                            clone.perceptrons[i] = perceptrons[i];

                        return clone;
                    }

                    public override float Forward(float input)
                    {
                        switch (func)
                        {
                            case ActivationFunc.Linear:
                                return input;
                            case ActivationFunc.Tanh:
                                return (2 / (1 + BTMath.singleton.Exp(-2 * input))) - 1;
                            case ActivationFunc.Sigmoid:
                                return 1 / (1 + BTMath.singleton.Exp(-input));
                            case ActivationFunc.Squared:
                                return input * input;
                            case ActivationFunc.Absolute:
                                return input < 0 ? -input : input;
                            case ActivationFunc.ReLU:
                                return input < 0 ? 0 : input;
                            case ActivationFunc.Inverse:
                                return -input;
                            default:
                                return input;
                        }
                    }
                }
                #endregion
            }

            public interface INeuralNetwork : IDeepClonable<INeuralNetwork>, ICustomSerializable
            {
                float[] Predict(params float[] inputs);

                void Crossover(INeuralNetwork network);

                void Mutate(float mutationRate, INumMutator mutator = null);

                abstract string ICustomSerializable.ToJson();

                abstract T ICustomSerializable.FromJson<T>(string json);
            }

            #endregion

            #region Number Mutator

            public class GaussianAdditiveNumMutator : INumMutator
            {
                public float stdev;

                public GaussianAdditiveNumMutator(float stdev)
                {
                    this.stdev = stdev;
                }

                public float Mutate(float value)
                {
                    return Randomizer.singleton.SignedGaussian(value, stdev);
                }
            }

            public class AdditiveNumMutator : INumMutator
            {
                public float additiveRange;

                public AdditiveNumMutator(float additiveRange)
                {
                    this.additiveRange = additiveRange;
                }

                public float Mutate(float value)
                {
                    return value + Randomizer.singleton.RandomRange(-additiveRange, additiveRange);
                }
            }

            public interface INumMutator
            {
                float Mutate(float value);
            }

            #endregion

            [Serializable]
            public class InnovationList<TValue> : IDictionary<int, List<TValue>>, IDeepClonable<InnovationList<TValue>> where TValue : IDeepClonable<TValue>, InnovationListItem
            {
                public event Action<int, TValue> onAdd, onRemove;

                Dictionary<int, List<TValue>> dic;

                public int Count => dic.Count;

                public ICollection<int> Keys => dic.Keys;

                public ICollection<List<TValue>> Values => dic.Values;

                public bool IsReadOnly => ((ICollection<KeyValuePair<int, List<TValue>>>)dic).IsReadOnly;

                public InnovationList()
                {
                    dic = new Dictionary<int, List<TValue>>();
                }

                public int FindValueNum()
                {
                    int valueCount = 0;

                    foreach (int innovationNum in Keys)
                        for (int i = 0, count = dic[innovationNum].Count; i < count; i++)
                            valueCount++;

                    return valueCount;
                }

                public TValue PickRandom(int fromInnovation = 0, int toInnovation = -1)
                {
                    if (toInnovation == -1)
                        return PickRandom(dic[dic.ElementAt(Randomizer.singleton.RandomRange(fromInnovation, dic.Count - 1)).Key]);
                    else
                        return PickRandom(dic[dic.ElementAt(Randomizer.singleton.RandomRange(fromInnovation, dic.Count - 1)).Key]);
                }

                public TValue PickRandom(out int innovationNum, int fromInnovation = 0, int toInnovation = -1)
                {
                    if (toInnovation == -1)
                        return PickRandom(dic[innovationNum = dic.ElementAt(Randomizer.singleton.RandomRange(fromInnovation, dic.Count - 1)).Key]);
                    else
                        return PickRandom(dic[innovationNum = dic.ElementAt(Randomizer.singleton.RandomRange(fromInnovation, dic.Count - 1)).Key]);
                }

                public TValue PickRandom(out int innovationNum, out int innerListIndex, int fromInnovation = 0, int toInnovation = -1)
                {
                    if (toInnovation == -1)
                        return PickRandom(dic[innovationNum = dic.ElementAt(Randomizer.singleton.RandomRange(fromInnovation, dic.Count - 1)).Key], out innerListIndex);
                    else
                        return PickRandom(dic[innovationNum = dic.ElementAt(Randomizer.singleton.RandomRange(fromInnovation, toInnovation)).Key], out innerListIndex);
                }

                static TValue PickRandom(List<TValue> list) => list[Randomizer.singleton.RandomRange(0, list.Count - 1)];

                static TValue PickRandom(List<TValue> list, out int index)
                {
                    index = Randomizer.singleton.RandomRange(0, list.Count - 1);


                    return list[index];
                }

                public List<TValue> this[int key] { get => dic[key]; set => dic[key] = value; }

                public void Add(int innovationNum, TValue value)
                {
                    if (ContainsKey(innovationNum))
                    {
                        onAdd?.Invoke(innovationNum, value);
                        value.OnAdd(innovationNum);
                        this[innovationNum].Add(value);
                    }
                    else
                        Add(innovationNum, new List<TValue>() { value });
                }

                public bool Remove(int innovationNum, TValue value)
                {
                    if (ContainsKey(innovationNum))
                    {
                        if (this[innovationNum].Remove(value))
                        {
                            onRemove?.Invoke(innovationNum, value);
                            value.OnRemove(innovationNum);
                            return true;
                        }
                    }

                    return false;
                }

                public void RemoveAt(int innovationNum, int innerListIndex)
                {
                    TValue removeValue = this[innovationNum][innerListIndex];
                    this[innovationNum].RemoveAt(innerListIndex);
                    onRemove?.Invoke(innovationNum, removeValue);
                    removeValue.OnRemove(innovationNum);
                }

                public bool ContainsKey(int key)
                {
                    return dic.ContainsKey(key);
                }

                public void Add(int innovationNum, List<TValue> values)
                {
                    if (ContainsKey(innovationNum))
                    {
                        for (int i = 0; i < values.Count; i++)
                            this.Add(innovationNum, values[i]);
                    }
                    else
                    {
                        dic.Add(innovationNum, values);

                        for (int i = 0; i < values.Count; i++)
                        {
                            onAdd?.Invoke(innovationNum, values[i]);
                            values[i].OnAdd(innovationNum);
                        }
                    }
                }

                public bool Remove(int innovationNum)
                {
                    TValue[] removeValues = this[innovationNum].ToArray();

                    if (!dic.Remove(innovationNum))
                        return false;

                    for (int i = 0; i < removeValues.Length; i++)
                    {
                        onRemove?.Invoke(innovationNum, removeValues[i]);
                        removeValues[i].OnRemove(innovationNum);
                    }

                    return true;
                }

                public bool TryGetValue(int innovationNum, out List<TValue> values)
                {
                    return dic.TryGetValue(innovationNum, out values);
                }

                public void Add(KeyValuePair<int, List<TValue>> item)
                {
                    Add(item.Key, item.Value);
                }

                public void Clear()
                {
                    foreach (int innovationNum in dic.Keys)
                        for (int i = 0, count = dic[innovationNum].Count; i < count; i++)
                        {
                            onRemove?.Invoke(innovationNum, dic[innovationNum][i]);
                            dic[innovationNum][i].OnRemove(innovationNum);
                        }

                    ((ICollection<KeyValuePair<int, List<TValue>>>)dic).Clear();
                }

                public bool Contains(KeyValuePair<int, List<TValue>> item)
                {
                    return ((ICollection<KeyValuePair<int, List<TValue>>>)dic).Contains(item);
                }

                public void CopyTo(KeyValuePair<int, List<TValue>>[] array, int arrayIndex)
                {
                    ((ICollection<KeyValuePair<int, List<TValue>>>)dic).CopyTo(array, arrayIndex);
                }

                public bool Remove(KeyValuePair<int, List<TValue>> item)
                {
                    if (!((ICollection<KeyValuePair<int, List<TValue>>>)dic).Remove(item))
                        return false;

                    for (int i = 0; i < item.Value.Count; i++)
                    {
                        onAdd?.Invoke(item.Key, item.Value[i]);
                        item.Value[i].OnRemove(item.Key);
                    }

                    return true;
                }

                public IEnumerator<KeyValuePair<int, List<TValue>>> GetEnumerator()
                {
                    return ((IEnumerable<KeyValuePair<int, List<TValue>>>)dic).GetEnumerator();
                }

                IEnumerator IEnumerable.GetEnumerator()
                {
                    return ((IEnumerable)dic).GetEnumerator();
                }

                public InnovationList<TValue> DeepClone()
                {
                    InnovationList<TValue> clone = new InnovationList<TValue>();

                    foreach (int key in Keys)
                    {
                        clone[key] = new List<TValue>();

                        for (int i = 0; i < this[key].Count; i++)
                            clone[key].Add(this[key][i].DeepClone());
                    }

                    clone.onAdd = (Action<int, TValue>)onAdd.Clone();

                    return clone;
                }
            }
        }
    }

    #region Randmizer

    public abstract class Randomizer
    {
        static Randomizer _singleton;
        public static Randomizer singleton
        {
            get
            {
                if (_singleton == null)
                    _singleton = new UnityRandomizer();

                return _singleton;
            }
            set => _singleton = value;
        }

        public abstract float Gaussian(float mean, float stdev);

        public abstract float SignedGaussian(float mean, float stdev);

        public abstract int RandomRange(int min, int max);

        public abstract float RandomRange(float min, float max);

        /// <summary>
        /// 
        /// </summary>
        /// <returns>A float of a value between 0.0f and 1.0f</returns>
        public abstract float RandomFloat();

        public abstract bool RandomBool();
    }

    public class UnityRandomizer : Randomizer
    {
        public override float Gaussian(float mean, float stdev)
        {
            return
                BTMath.singleton.Sqrt(-2 * BTMath.singleton.Log(RandomFloat())) * BTMath.singleton.Cos(2 * BTMath.singleton.PI * RandomFloat()) * stdev + mean;
        }

        public override float SignedGaussian(float mean, float stdev)
        {
            return
                (RandomRange(0, 1) * 2 - 1) * BTMath.singleton.Sqrt(-2 * BTMath.singleton.Log(RandomFloat())) * BTMath.singleton.Cos(2 * BTMath.singleton.PI * RandomFloat()) * stdev + mean;
        }

        public override bool RandomBool()
        {
            return (RandomRange(0, 1) == 1) ? true : false;
        }

        public override float RandomFloat()
        {
            return UnityEngine.Random.Range(0f, 1f);
        }

        public override int RandomRange(int min, int max)
        {
            return UnityEngine.Random.Range(min, max + 1);
        }

        public override float RandomRange(float min, float max)
        {
            return UnityEngine.Random.Range(min, max);
        }
    }

    public class ConsoleRandomizer : Randomizer
    {
        System.Random rand;

        public ConsoleRandomizer()
        {
            rand = new System.Random();
        }

        public override float Gaussian(float mean, float stdev)
        {
            // SQRT( -2*LN(RAND()) ) * COS( 2*PI()*RAND() ) * StdDev + Mean

            return (float)(
                Math.Sqrt(-2 * Math.Log(rand.NextDouble())) * Math.Cos(2 * Math.PI * rand.NextDouble()) * stdev + mean
            );
        }

        public override float SignedGaussian(float mean, float stdev)
        {
            // SQRT( -2*LN(RAND()) ) * COS( 2*PI()*RAND() ) * StdDev + Mean

            return (float)(
                (RandomRange(0, 1) * 2 - 1) * Math.Sqrt(-2 * Math.Log(rand.NextDouble())) * Math.Cos(2 * Math.PI * rand.NextDouble()) * stdev + mean
            );
        }

        public override int RandomRange(int min, int max)
        {
            return rand.Next(min, max + 1);
        }

        public override float RandomRange(float min, float max)
        {
            return (float)(min + (max - min) * rand.NextDouble());
        }

        public override float RandomFloat()
        {
            return (float)rand.NextDouble();
        }

        public override bool RandomBool()
        {
            return RandomRange(0, 1) == 1 ? true : false;
        }
    }

    #endregion

    #region Math

    public interface BTMath
    {
        static BTMath _singleton;
        public static BTMath singleton
        {
            get
            {
                if (_singleton == null)
                    _singleton = new UnityMath();

                return _singleton;
            }
            set => _singleton = value;
        }

        public abstract float PI { get; }

        public abstract float Pow(float value, float power);

        public abstract float Exp(float power);

        public abstract float Sqrt(float power);

        public abstract float Log(float power);

        public abstract float Log(float power, float baseNum);

        public abstract float Cos(float angle);

        public abstract float Sin(float angle);
    }

    public class UnityMath : BTMath
    {
        public float PI => Mathf.PI;

        public float Pow(float value, float power) => Mathf.Pow(value, power);

        public float Exp(float power) => Mathf.Exp(power);

        public float Sqrt(float power) => Mathf.Sqrt(power);

        public float Log(float power) => Mathf.Log(power);

        public float Log(float power, float baseNum) => Mathf.Log(power, baseNum);

        public float Cos(float angle) => Mathf.Cos(angle * Mathf.Deg2Rad);

        public float Sin(float angle) => Mathf.Sin(angle * Mathf.Deg2Rad);
    }

    public class ConsoleMath : BTMath
    {
        public float PI => (float)Math.PI;

        public float Pow(float value, float power) => (float)Math.Pow(value, power);

        public float Exp(float power) => (float)Math.Exp(power);

        public float Sqrt(float power) => (float)Math.Sqrt(power);

        public float Log(float power) => (float)Math.Log(power);

        public float Log(float power, float baseNum) => (float)Math.Log(power, baseNum);

        public float Cos(float angle) => (float)Math.Cos(angle * (Math.PI / 180));

        public float Sin(float angle) => (float)Math.Sin(angle * (Math.PI / 180));
    }

    #endregion

    public interface ICustomSerializable
    {
        string ToJson();

        T FromJson<T>(string json);
    }

    public interface IDeepClonable<T>
    {
        T DeepClone();
    }
}