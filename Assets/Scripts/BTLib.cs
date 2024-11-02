using BTLib.AI.RL;
using BTLib.Utility;
using System.Collections.Generic;
using System.IO;
using System;
using System.Linq;
using UnityEditor.PackageManager.UI;

namespace BTLib
{
    namespace AI
    {
        public class ForwardResult
        {
            public float[][][] layerInputs;
            public float[][] outputs;

            public ForwardResult(float[][][] layerInputs, float[][] outputs)
            {
                this.layerInputs = layerInputs;
                this.outputs = outputs;
            }
        }

        public interface INeuralNetwork : IPolicy
        {
            int InDim { get; }
            int OutDim { get; }
            Optimizer Optimizer { get; }
            Layer[] Layers { get; }
            WeightMatrix[] Weights { get; }
            ForwardResult Log { get; set; }

            float[] IPolicy.Forward(float[] obs)
            {
                Log = ForwardLog(obs);
                return Log.outputs[0];
            }

            void IPolicy.Update(float[] loss)
            {
                Backward(loss, Log);
            }

            ForwardResult ForwardLog(float[] inputs);

            /// <summary>
            /// Input batch forwarding
            /// </summary>
            ForwardResult ForwardLog(float[][] inputs);

            /// <summary>
            /// Backpropagates and updates weights, biases
            /// </summary>
            void Backward(float[] loss, ForwardResult forwardLog);

            /// <summary>
            /// Backpropagates and updates weights, biases in batches
            /// </summary>
            void Backward(float[][] loss, ForwardResult forwardLog);

            /// <summary>
            /// </summary>
            /// <param name="func">Takes current weight as parameter and returns a new weight</param>
            void WeightAssignForEach(Func<float, int, int, float> func);

            /// <summary>
            /// </summary>
            /// <param name="func">Takes current bias as parameter and returns a new bias</param>
            void BiasAssignForEach(Func<float, int, float> func);
        }

        public class DenseNeuralNetwork : INeuralNetwork
        {
            public int InDim { get; private set; }
            public int OutDim { get; private set; }
            public Optimizer Optimizer { get; private set; }
            public Layer[] Layers { get; private set; }
            public WeightMatrix[] Weights { get; private set; }

            public ForwardResult Log { get; set; }

            public DenseNeuralNetwork(DenseNeuralNetworkBuilder builder, float learningRate, bool disposeAfterwards = true) : base()
            {
                Tuple<Layer[], WeightMatrix[]> bundle = builder.Build();

                this.Layers = bundle.Item1;
                this.Weights = bundle.Item2;
                this.Optimizer = new SGD(learningRate);
                this.InDim = Layers[0].dim;
                this.OutDim = Layers[Layers.LongLength - 1].dim;

                foreach (var layer in Layers)
                    layer.Build(this);

                foreach (var weight in Weights)
                    weight.Build(this);

                Optimizer.Init(this);

                if (disposeAfterwards)
                    builder.Dispose();
            }

            public DenseNeuralNetwork(DenseNeuralNetworkBuilder builder, Optimizer optimizer, bool disposeAfterwards = true) : base()
            {
                Tuple<Layer[], WeightMatrix[]> bundle = builder.Build();

                this.Layers = bundle.Item1;
                this.Weights = bundle.Item2;
                this.Optimizer = optimizer;
                this.InDim = Layers[0].dim;
                this.OutDim = Layers[Layers.LongLength - 1].dim;

                foreach (var layer in Layers)
                    layer.Build(this);

                foreach (var weight in Weights)
                    weight.Build(this);

                optimizer.Init(this);

                if (disposeAfterwards)
                    builder.Dispose();
            }

            public void WeightAssignForEach(Func<float, int, int, float> func)
            {
                for (int i = 0; i < Weights.LongLength; i++)
                    Weights[i].AssignForEach((inIndex, outIndex, weight) => func(weight, Weights[i].inDim, Weights[i].outDim));
            }

            public void BiasAssignForEach(Func<float, int, float> func)
            {
                for (int i = 0; i < Layers.LongLength; i++)
                    for (int j = 0; j < Layers[i].dim; j++)
                        Layers[i].SetBias(j, func(Layers[i].GetBias(j), Layers[i].dim));
            }

            public void Backward(float[] loss, ForwardResult forwardLog)
            {
                float[][] temp = new float[1][];
                temp[0] = loss;

                Layers[Layers.Length - 1].GradientDescent(ref temp, forwardLog, Optimizer);

                for (int i = Layers.Length - 2; i > -1; i--)
                {
                    Weights[i].GradientDescent(ref temp, forwardLog, Optimizer);
                    Layers[i].GradientDescent(ref temp, forwardLog, Optimizer);
                }
            }

            public void Backward(float[][] loss, ForwardResult forwardLog)
            {
                Layers[Layers.Length - 1].GradientDescent(ref loss, forwardLog, Optimizer);

                for (int i = Layers.Length - 2; i > -1; i--)
                {
                    Weights[i].GradientDescent(ref loss, forwardLog, Optimizer);
                    Layers[i].GradientDescent(ref loss, forwardLog, Optimizer);
                }
            }

            public ForwardResult ForwardLog(float[] inputs)
            {
                float[][][] layerInputs = new float[Layers.LongLength][][];
                float[] outputs = ForwardLayers(inputs, Layers.Length - 1, 0, ref layerInputs);

                return new ForwardResult(layerInputs, new float[][] { outputs });
            }

            public ForwardResult ForwardLog(float[][] inputs)
            {
                float[][][] layerInputs = new float[Layers.LongLength][][];
                float[][] outputs = ForwardLayers(inputs, Layers.Length - 1, 0, ref layerInputs);

                return new ForwardResult(layerInputs, outputs);
            }

            float[] ForwardLayers(float[] inputs, int toLayer, int fromLayer, ref float[][][] layerInputs)
            {
                layerInputs[toLayer] = new float[1][];

                if (fromLayer < toLayer)
                    layerInputs[toLayer][0] = Weights[toLayer - 1].Forward(ForwardLayers(inputs, toLayer - 1, fromLayer, ref layerInputs));
                else
                    layerInputs[toLayer][0] = inputs;

                return Layers[toLayer].Forward(layerInputs[toLayer][0]);
            }

            float[][] ForwardLayers(float[][] inputs, int toLayer, int fromLayer, ref float[][][] layerInputs)
            {
                if (fromLayer < toLayer)
                    layerInputs[toLayer] = Weights[toLayer - 1].Forward(ForwardLayers(inputs, toLayer - 1, fromLayer, ref layerInputs));
                else
                    layerInputs[toLayer] = inputs;

                return Layers[toLayer].Forward(layerInputs[toLayer]);
            }

        }

        public class DenseNeuralNetworkBuilder : INeuralNetworkBuilder, IDisposable
        {
            public List<Layer> layers;

            public DenseNeuralNetworkBuilder(int inputDim)
            {
                layers = new List<Layer>();

                layers.Add(new Layer(inputDim, false));
            }

            public void NewLayers(params Layer[] dims)
            {
                foreach (Layer dim in dims)
                    NewLayer(dim);
            }

            public void NewLayer(int dim)
            {
                layers.Add(new Layer(dim));
            }

            public void NewLayer(Layer layer)
            {
                layers.Add(layer);
            }

            public Tuple<Layer[], WeightMatrix[]> Build()
            {
                WeightMatrix[] weights = new WeightMatrix[layers.Count - 1];

                for (int i = 1; i < layers.Count; i++)
                {
                    if (layers[i - 1] is ForwardLayer && ((ForwardLayer)layers[i - 1]).port != ForwardLayer.ForwardPort.In)
                        weights[i - 1] = layers[i - 1].GenerateWeightMatrix();
                    else
                        weights[i - 1] = layers[i].GenerateWeightMatrix();
                }

                return (layers.ToArray(), weights).ToTuple();
            }

            public void Dispose()
            {
                GC.SuppressFinalize(this);
            }
        }

        public interface INeuralNetworkBuilder : IDisposable
        {
            abstract Tuple<Layer[], WeightMatrix[]> Build();
        }

        #region Optimizer

        public interface IBatchNormOptimizable
        {
            public Dictionary<int, int> bnIndexLookup { get; }

            public float GammaUpdate(int layerIndex, float gradient);

            public float BetaUpdate(int layerIndex, float gradient);
        }

        public abstract class Optimizer
        {
            public DenseNeuralNetwork network;
            public float weightDecay;

            public Optimizer(float weightDecay = 0)
            {
                this.weightDecay = weightDecay;
            }

            public virtual void Init(DenseNeuralNetwork network)
            {
                this.network = network;
            }

            public abstract float WeightUpdate(int weightsIndex, int inIndex, int outIndex, float gradient);

            public abstract float BiasUpdate(int layerIndex, int perceptron, float gradient);
        }

        public class SGD : Optimizer, IBatchNormOptimizable
        {
            public float learningRate;
            public Dictionary<int, int> bnIndexLookup { get; private set; }

            public SGD(float learningRate, float weightDecay = 0) : base(weightDecay)
            {
                this.learningRate = learningRate;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;
            }

            public override float WeightUpdate(int weightsIndex, int inIndex, int outIndex, float gradient)
            {
                return network.Weights[weightsIndex].GetWeight(inIndex, outIndex) * (1 - weightDecay) - gradient * learningRate;
            }

            public override float BiasUpdate(int layerIndex, int perceptron, float gradient)
            {
                return network.Layers[layerIndex].GetBias(perceptron) - gradient * learningRate;
            }

            public float GammaUpdate(int layerIndex, float gradient)
            {
                return ((BatchNormLayer)network.Layers[layerIndex]).gamma - gradient * learningRate;
            }

            public float BetaUpdate(int layerIndex, float gradient)
            {
                return ((BatchNormLayer)network.Layers[layerIndex]).beta - gradient * learningRate;
            }
        }

        public class Momentum : Optimizer, IBatchNormOptimizable
        {
            public float[][][] weightMomentum;
            public float[][] biasMomentum;
            public float learningRate, beta;

            public Dictionary<int, int> bnIndexLookup { get; private set; }
            public float[] gammaMomentum, betaMomentum;

            public Momentum(float beta = 0.9f, float learningRate = 0.01f, float weightDecay = 0) : base(weightDecay)
            {
                this.learningRate = learningRate;
                this.beta = beta;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;

                weightMomentum = new float[network.Weights.Length][][];
                for (int i = 0; i < network.Weights.Length; i++)
                {
                    weightMomentum[i] = new float[network.Weights[i].outDim][];
                    for (int j = 0; j < network.Weights[i].outDim; j++)
                    {
                        weightMomentum[i][j] = new float[network.Weights[i].inDim];
                        for (int k = 0; k < network.Weights[i].inDim; k++)
                            weightMomentum[i][j][k] = 0.000001f; // epsilon = 10^-6
                    }
                }

                bnIndexLookup = new Dictionary<int, int>();

                biasMomentum = new float[network.Layers.Length][];
                for (int i = 0; i < network.Layers.Length; i++)
                {
                    biasMomentum[i] = new float[network.Layers[i].dim];
                    for (int j = 0; j < network.Layers[i].dim; j++)
                        biasMomentum[i][j] = 0.000001f; // epsilon = 10^-6

                    if (network.Layers[i] is BatchNormLayer)
                        bnIndexLookup.Add(i, bnIndexLookup.Count);
                }

                gammaMomentum = new float[bnIndexLookup.Count];
                betaMomentum = new float[bnIndexLookup.Count];
            }

            public override float WeightUpdate(int weightsIndex, int inIndex, int outIndex, float gradient)
            {
                weightMomentum[weightsIndex][outIndex][inIndex] =
                    beta * weightMomentum[weightsIndex][outIndex][inIndex] +
                    (1 - beta) * (gradient + weightDecay * network.Weights[weightsIndex].GetWeight(inIndex, outIndex));

                return network.Weights[weightsIndex].GetWeight(inIndex, outIndex) - learningRate * weightMomentum[weightsIndex][outIndex][inIndex];
            }

            public override float BiasUpdate(int layerIndex, int perceptron, float gradient)
            {
                biasMomentum[layerIndex][perceptron] = beta * biasMomentum[layerIndex][perceptron] + (1 - beta) * gradient;

                return network.Layers[layerIndex].GetBias(perceptron) - learningRate * biasMomentum[layerIndex][perceptron];
            }

            public float GammaUpdate(int layerIndex, float gradient)
            {
                gammaMomentum[bnIndexLookup[layerIndex]] = beta * gammaMomentum[bnIndexLookup[layerIndex]] + (1 - beta) * gradient;

                return ((BatchNormLayer)network.Layers[layerIndex]).gamma - learningRate * gammaMomentum[bnIndexLookup[layerIndex]];
            }

            public float BetaUpdate(int layerIndex, float gradient)
            {
                betaMomentum[bnIndexLookup[layerIndex]] = beta * betaMomentum[bnIndexLookup[layerIndex]] + (1 - beta) * gradient;

                return ((BatchNormLayer)network.Layers[layerIndex]).beta - learningRate * betaMomentum[bnIndexLookup[layerIndex]];
            }
        }

        public class RMSprop : Optimizer, IBatchNormOptimizable
        {
            public float[][][] accumWeightGrad;
            public float[][] accumBiasGrad;
            public float learningRate, beta;

            public Dictionary<int, int> bnIndexLookup { get; private set; }
            public float[] accumGammaGrad, accumBetaGrad;

            public RMSprop(float beta = 0.99f, float learningRate = 0.01f, float weightDecay = 0) : base(weightDecay)
            {
                this.learningRate = learningRate;
                this.beta = beta;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;

                accumWeightGrad = new float[network.Weights.Length][][];
                for (int i = 0; i < network.Weights.Length; i++)
                {
                    accumWeightGrad[i] = new float[network.Weights[i].outDim][];
                    for (int j = 0; j < network.Weights[i].outDim; j++)
                    {
                        accumWeightGrad[i][j] = new float[network.Weights[i].inDim];
                        for (int k = 0; k < network.Weights[i].inDim; k++)
                            accumWeightGrad[i][j][k] = 0.000001f; // epsilon = 10^-6
                    }
                }

                bnIndexLookup = new Dictionary<int, int>();

                accumBiasGrad = new float[network.Layers.Length][];
                for (int i = 0; i < network.Layers.Length; i++)
                {
                    accumBiasGrad[i] = new float[network.Layers[i].dim];
                    for (int j = 0; j < network.Layers[i].dim; j++)
                        accumBiasGrad[i][j] = 0.000001f; // epsilon = 10^-6

                    if (network.Layers[i] is BatchNormLayer)
                        bnIndexLookup.Add(i, bnIndexLookup.Count);
                }

                accumGammaGrad = new float[bnIndexLookup.Count];
                accumBetaGrad = new float[bnIndexLookup.Count];
            }

            public override float WeightUpdate(int weightsIndex, int inIndex, int outIndex, float gradient)
            {
                accumWeightGrad[weightsIndex][outIndex][inIndex] =
                    beta * accumWeightGrad[weightsIndex][outIndex][inIndex] +
                    (1 - beta) * (gradient * gradient + weightDecay * network.Weights[weightsIndex].GetWeight(inIndex, outIndex));

                return network.Weights[weightsIndex].GetWeight(inIndex, outIndex) - (learningRate * gradient / MathF.Sqrt(accumWeightGrad[weightsIndex][outIndex][inIndex]));
            }

            public override float BiasUpdate(int layerIndex, int perceptron, float gradient)
            {
                accumBiasGrad[layerIndex][perceptron] = beta * accumBiasGrad[layerIndex][perceptron] + (1 - beta) * gradient * gradient;

                return network.Layers[layerIndex].GetBias(perceptron) - (learningRate * gradient / MathF.Sqrt(accumBiasGrad[layerIndex][perceptron]));
            }

            public float GammaUpdate(int layerIndex, float gradient)
            {
                accumGammaGrad[bnIndexLookup[layerIndex]] = beta * accumGammaGrad[bnIndexLookup[layerIndex]] + (1 - beta) * gradient * gradient;

                return ((BatchNormLayer)network.Layers[layerIndex]).gamma - (learningRate * gradient / MathF.Sqrt(accumGammaGrad[bnIndexLookup[layerIndex]]));
            }

            public float BetaUpdate(int layerIndex, float gradient)
            {
                accumBetaGrad[bnIndexLookup[layerIndex]] = beta * accumBetaGrad[bnIndexLookup[layerIndex]] + (1 - beta) * gradient * gradient;

                return ((BatchNormLayer)network.Layers[layerIndex]).beta - (learningRate * gradient / MathF.Sqrt(accumBetaGrad[bnIndexLookup[layerIndex]]));
            }
        }

        public class Adam : Optimizer, IBatchNormOptimizable
        {
            public float[][][] accumWeightGrad, weightMomentum;
            public float[][] accumBiasGrad, biasMomentum;
            public float learningRate, beta1, beta2;

            public Dictionary<int, int> bnIndexLookup { get; private set; }
            public float[] accumGammaGrad, gammaMomentum, accumBetaGrad, betaMomentum;

            public Adam(float beta1 = 0.9f, float beta2 = 0.99f, float learningRate = 0.01f, float weightDecay = 0) : base(weightDecay)
            {
                this.learningRate = learningRate;
                this.beta1 = beta1;
                this.beta2 = beta2;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;

                weightMomentum = new float[network.Weights.Length][][];
                accumWeightGrad = new float[network.Weights.Length][][];
                for (int i = 0; i < network.Weights.Length; i++)
                {
                    weightMomentum[i] = new float[network.Weights[i].outDim][];
                    accumWeightGrad[i] = new float[network.Weights[i].outDim][];
                    for (int j = 0; j < network.Weights[i].outDim; j++)
                    {
                        weightMomentum[i][j] = new float[network.Weights[i].inDim];
                        accumWeightGrad[i][j] = new float[network.Weights[i].inDim];
                        for (int k = 0; k < network.Weights[i].inDim; k++)
                        {
                            weightMomentum[i][j][k] = 0.000001f; // epsilon = 10^-6
                            accumWeightGrad[i][j][k] = 0.000001f; // epsilon = 10^-6
                        }
                    }
                }

                bnIndexLookup = new Dictionary<int, int>();

                biasMomentum = new float[network.Layers.Length][];
                accumBiasGrad = new float[network.Layers.Length][];
                for (int i = 0; i < network.Layers.Length; i++)
                {
                    biasMomentum[i] = new float[network.Layers[i].dim];
                    accumBiasGrad[i] = new float[network.Layers[i].dim];
                    for (int j = 0; j < network.Layers[i].dim; j++)
                    {
                        biasMomentum[i][j] = 0.000001f; // epsilon = 10^-6
                        accumBiasGrad[i][j] = 0.000001f; // epsilon = 10^-6
                    }

                    if (network.Layers[i] is BatchNormLayer)
                        bnIndexLookup.Add(i, bnIndexLookup.Count);
                }

                gammaMomentum = new float[bnIndexLookup.Count];
                accumGammaGrad = new float[bnIndexLookup.Count];
                betaMomentum = new float[bnIndexLookup.Count];
                accumBetaGrad = new float[bnIndexLookup.Count];
            }

            public override float WeightUpdate(int weightsIndex, int inIndex, int outIndex, float gradient)
            {
                weightMomentum[weightsIndex][outIndex][inIndex] =
                    beta1 * weightMomentum[weightsIndex][outIndex][inIndex] +
                    (1 - beta1) * (gradient + weightDecay * network.Weights[weightsIndex].GetWeight(inIndex, outIndex));

                accumWeightGrad[weightsIndex][outIndex][inIndex] =
                    beta2 * accumWeightGrad[weightsIndex][outIndex][inIndex] +
                    (1 - beta2) * (gradient * gradient + weightDecay * network.Weights[weightsIndex].GetWeight(inIndex, outIndex));

                return network.Weights[weightsIndex].GetWeight(inIndex, outIndex) - (learningRate * weightMomentum[weightsIndex][outIndex][inIndex] / MathF.Sqrt(accumWeightGrad[weightsIndex][outIndex][inIndex]));
            }

            public override float BiasUpdate(int layerIndex, int perceptron, float gradient)
            {
                biasMomentum[layerIndex][perceptron] = beta1 * biasMomentum[layerIndex][perceptron] + (1 - beta1) * gradient;
                accumBiasGrad[layerIndex][perceptron] = beta2 * accumBiasGrad[layerIndex][perceptron] + (1 - beta2) * gradient * gradient;

                return network.Layers[layerIndex].GetBias(perceptron) - (learningRate * biasMomentum[layerIndex][perceptron] / MathF.Sqrt(accumBiasGrad[layerIndex][perceptron]));
            }

            public float GammaUpdate(int layerIndex, float gradient)
            {
                gammaMomentum[bnIndexLookup[layerIndex]] = beta1 * gammaMomentum[bnIndexLookup[layerIndex]] + (1 - beta1) * gradient;
                accumGammaGrad[bnIndexLookup[layerIndex]] = beta2 * accumGammaGrad[bnIndexLookup[layerIndex]] + (1 - beta2) * gradient * gradient;

                return ((BatchNormLayer)network.Layers[layerIndex]).gamma - (learningRate * gammaMomentum[bnIndexLookup[layerIndex]] / MathF.Sqrt(accumGammaGrad[bnIndexLookup[layerIndex]]));
            }

            public float BetaUpdate(int layerIndex, float gradient)
            {
                betaMomentum[bnIndexLookup[layerIndex]] = beta1 * betaMomentum[bnIndexLookup[layerIndex]] + (1 - beta1) * gradient;
                accumBetaGrad[bnIndexLookup[layerIndex]] = beta2 * accumBetaGrad[bnIndexLookup[layerIndex]] + (1 - beta2) * gradient * gradient;

                return ((BatchNormLayer)network.Layers[layerIndex]).beta - (learningRate * betaMomentum[bnIndexLookup[layerIndex]] / MathF.Sqrt(accumBetaGrad[bnIndexLookup[layerIndex]]));
            }
        }

        public class AdaGrad : Optimizer, IBatchNormOptimizable
        {
            public float[][][] accumWeightGrad;
            public float[][] accumBiasGrad;
            public float eta;

            public Dictionary<int, int> bnIndexLookup { get; private set; }
            public float[] accumGammaGrad, accumBetaGrad;

            public AdaGrad(float eta = 0.01f, float weightDecay = 0) : base(weightDecay)
            {
                this.eta = eta;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;

                accumWeightGrad = new float[network.Weights.Length][][];
                for (int i = 0; i < network.Weights.Length; i++)
                {
                    accumWeightGrad[i] = new float[network.Weights[i].outDim][];
                    for (int j = 0; j < network.Weights[i].outDim; j++)
                    {
                        accumWeightGrad[i][j] = new float[network.Weights[i].inDim];
                        for (int k = 0; k < network.Weights[i].inDim; k++)
                            accumWeightGrad[i][j][k] = 0.000001f; // epsilon = 10^-6
                    }
                }

                bnIndexLookup = new Dictionary<int, int>();

                accumBiasGrad = new float[network.Layers.Length][];
                for (int i = 0; i < network.Layers.Length; i++)
                {
                    accumBiasGrad[i] = new float[network.Layers[i].dim];
                    for (int j = 0; j < network.Layers[i].dim; j++)
                        accumBiasGrad[i][j] = 0.000001f; // epsilon = 10^-6

                    if (network.Layers[i] is BatchNormLayer)
                        bnIndexLookup.Add(i, bnIndexLookup.Count);
                }

                accumGammaGrad = new float[bnIndexLookup.Count];
                accumBetaGrad = new float[bnIndexLookup.Count];
            }

            public override float WeightUpdate(int weightsIndex, int inIndex, int outIndex, float gradient)
            {
                accumWeightGrad[weightsIndex][outIndex][inIndex] += gradient * gradient;

                return (1 - weightDecay) * network.Weights[weightsIndex].GetWeight(inIndex, outIndex) - (eta / MathF.Sqrt(accumWeightGrad[weightsIndex][outIndex][inIndex])) * gradient;
            }

            public override float BiasUpdate(int layerIndex, int perceptron, float gradient)
            {
                accumBiasGrad[layerIndex][perceptron] += gradient * gradient;

                return network.Layers[layerIndex].GetBias(perceptron) - (eta / MathF.Sqrt(accumBiasGrad[layerIndex][perceptron])) * gradient;
            }

            public float GammaUpdate(int layerIndex, float gradient)
            {
                accumGammaGrad[bnIndexLookup[layerIndex]] += gradient * gradient;

                return ((BatchNormLayer)network.Layers[layerIndex]).gamma - (eta / MathF.Sqrt(accumGammaGrad[bnIndexLookup[layerIndex]])) * gradient;
            }

            public float BetaUpdate(int layerIndex, float gradient)
            {
                accumBetaGrad[bnIndexLookup[layerIndex]] += gradient * gradient;

                return ((BatchNormLayer)network.Layers[layerIndex]).beta - (eta / MathF.Sqrt(accumBetaGrad[bnIndexLookup[layerIndex]])) * gradient;
            }
        }

        public class AdaDelta : Optimizer, IBatchNormOptimizable
        {
            public float[][][] accumWeightGrad, accumRescaledWeightGrad;
            public float[][] accumBiasGrad, accumRescaledBiasGrad;
            public float rho;

            public Dictionary<int, int> bnIndexLookup { get; private set; }
            public float[] accumGammaGrad, accumBetaGrad, accumRescaledGammaGrad, accumRescaledBetaGrad;

            public AdaDelta(float rho = 0.9f, float weightDecay = 0) : base(weightDecay)
            {
                this.rho = rho;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;

                accumWeightGrad = new float[network.Weights.Length][][];
                accumRescaledWeightGrad = new float[network.Weights.Length][][];
                for (int i = 0; i < network.Weights.Length; i++)
                {
                    accumWeightGrad[i] = new float[network.Weights[i].outDim][];
                    accumRescaledWeightGrad[i] = new float[network.Weights[i].outDim][];
                    for (int j = 0; j < network.Weights[i].outDim; j++)
                    {
                        accumWeightGrad[i][j] = new float[network.Weights[i].inDim];
                        accumRescaledWeightGrad[i][j] = new float[network.Weights[i].inDim];
                        for (int k = 0; k < network.Weights[i].inDim; k++)
                        {
                            accumWeightGrad[i][j][k] = 0.000001f; // epsilon = 10^-6
                            accumRescaledWeightGrad[i][j][k] = 0.0000001f;
                        }
                    }
                }

                bnIndexLookup = new Dictionary<int, int>();

                accumBiasGrad = new float[network.Layers.Length][];
                accumRescaledBiasGrad = new float[network.Layers.Length][];
                for (int i = 0; i < network.Layers.Length; i++)
                {
                    accumBiasGrad[i] = new float[network.Layers[i].dim];
                    accumRescaledBiasGrad[i] = new float[network.Layers[i].dim];
                    for (int j = 0; j < network.Layers[i].dim; j++)
                    {
                        accumBiasGrad[i][j] = 0.000001f; // epsilon = 10^-6
                        accumRescaledBiasGrad[i][j] = 0.0000001f;
                    }

                    if (network.Layers[i] is BatchNormLayer)
                        bnIndexLookup.Add(i, bnIndexLookup.Count);
                }

                bool hasBatchNormLayer = false;
                foreach (Layer layer in network.Layers)
                    if (layer is BatchNormLayer)
                    {
                        hasBatchNormLayer = true;
                        break;
                    }

                if (!hasBatchNormLayer)
                    return;

                accumGammaGrad = new float[bnIndexLookup.Count];
                accumBetaGrad = new float[bnIndexLookup.Count];
                accumRescaledGammaGrad = new float[bnIndexLookup.Count];
                accumRescaledBetaGrad = new float[bnIndexLookup.Count];
            }

            public override float WeightUpdate(int weightsIndex, int inIndex, int outIndex, float gradient)
            {
                accumWeightGrad[weightsIndex][outIndex][inIndex] = rho * accumWeightGrad[weightsIndex][outIndex][inIndex] + (1 - rho) * gradient * gradient;

                float rescaledGrad =
                    MathF.Sqrt(accumRescaledWeightGrad[weightsIndex][outIndex][inIndex] / accumWeightGrad[weightsIndex][outIndex][inIndex]) * gradient +
                    weightDecay * network.Weights[weightsIndex].GetWeight(inIndex, outIndex);
                accumRescaledWeightGrad[weightsIndex][outIndex][inIndex] = rho * accumRescaledWeightGrad[weightsIndex][outIndex][inIndex] + (1 - rho) * rescaledGrad * rescaledGrad;

                return network.Weights[weightsIndex].GetWeight(inIndex, outIndex) - rescaledGrad;
            }

            public override float BiasUpdate(int layerIndex, int perceptron, float gradient)
            {
                accumBiasGrad[layerIndex][perceptron] = rho * accumBiasGrad[layerIndex][perceptron] + (1 - rho) * gradient * gradient;

                float rescaledGrad = MathF.Sqrt(accumRescaledBiasGrad[layerIndex][perceptron] / accumBiasGrad[layerIndex][perceptron]) * gradient;
                accumRescaledBiasGrad[layerIndex][perceptron] = rho * accumRescaledBiasGrad[layerIndex][perceptron] + (1 - rho) * rescaledGrad * rescaledGrad;

                return network.Layers[layerIndex].GetBias(perceptron) - rescaledGrad;
            }

            public float GammaUpdate(int layerIndex, float gradient)
            {
                accumGammaGrad[bnIndexLookup[layerIndex]] = rho * accumGammaGrad[bnIndexLookup[layerIndex]] + (1 - rho) * gradient * gradient;

                float rescaledGrad = MathF.Sqrt(accumRescaledGammaGrad[bnIndexLookup[layerIndex]] / accumGammaGrad[bnIndexLookup[layerIndex]]) * gradient;
                accumRescaledGammaGrad[bnIndexLookup[layerIndex]] = rho * accumRescaledGammaGrad[bnIndexLookup[layerIndex]] + (1 - rho) * rescaledGrad * rescaledGrad;

                return ((BatchNormLayer)network.Layers[layerIndex]).gamma - rescaledGrad;
            }

            public float BetaUpdate(int layerIndex, float gradient)
            {
                accumBetaGrad[bnIndexLookup[layerIndex]] = rho * accumBetaGrad[bnIndexLookup[layerIndex]] + (1 - rho) * gradient * gradient;

                float rescaledGrad = MathF.Sqrt(accumRescaledBetaGrad[bnIndexLookup[layerIndex]] / accumBetaGrad[bnIndexLookup[layerIndex]]) * gradient;
                accumRescaledBetaGrad[bnIndexLookup[layerIndex]] = rho * accumRescaledBetaGrad[bnIndexLookup[layerIndex]] + (1 - rho) * rescaledGrad * rescaledGrad;

                return ((BatchNormLayer)network.Layers[layerIndex]).beta - rescaledGrad;
            }
        }

        #endregion

        #region Layer

        public enum ActivationFunc
        {
            ReLU,
            Sigmoid,
            Tanh,
            NaturalLog,
            Exponential,
            Linear,
            Softmax,
            Custom
        }

        public class BatchNormLayer : ForwardLayer
        {
            public float gamma = 1, beta = 0;

            public BatchNormLayer(ForwardPort port) : base(ActivationFunc.Custom, port, false) { }

            public override float[][] Forward(float[][] inputs)
            {
                int sampleSize = inputs.Length;
                float[][] result = new float[sampleSize][];

                for (int sample = 0; sample < result.Length; sample++)
                    result[sample] = new float[dim];

                for (int i = 0; i < dim; i++)
                {
                    float mean = 0, variance = 0;

                    for (int sample = 0; sample < sampleSize; sample++)
                        mean += inputs[sample][i];
                    mean /= sampleSize;

                    for (int sample = 0; sample < sampleSize; sample++)
                        variance += MathF.Pow(inputs[sample][i] - mean, 2);
                    variance /= sampleSize;

                    for (int sample = 0; sample < result.Length; sample++)
                        result[sample][i] = Standardize(inputs[sample][i], mean, variance) * gamma + beta;
                }

                return result;
            }

            public override void GradientDescent(ref float[][] errors, ForwardResult log, Optimizer optimizer)
            {
                if (layerIndex == 0)
                    return;

                Layer prevLayer = network.Layers[layerIndex - 1];

                int sampleSize = errors.Length;

                float[] means = new float[dim];
                float[] variances = new float[dim];

                for (int i = 0; i < dim; i++)
                {
                    for (int sample = 0; sample < sampleSize; sample++)
                        means[i] += log.layerInputs[layerIndex][sample][i];
                    means[i] /= sampleSize;

                    for (int sample = 0; sample < sampleSize; sample++)
                        variances[i] += MathF.Pow(log.layerInputs[layerIndex][sample][i] - means[i], 2);
                    variances[i] /= sampleSize;
                    variances[i] += 0.000001f;
                }

                float[]
                    dbeta = new float[dim],
                    dgamma = new float[dim],
                    dvariances = new float[dim],
                    dmeans = new float[dim];

                for (int i = 0; i < dim; i++)
                {
                    for (int sample = 0; sample < sampleSize; sample++)
                    {
                        dbeta[i] += errors[sample][i];
                        dgamma[i] += errors[sample][i] * Standardize(log.layerInputs[layerIndex][sample][i], means[i], variances[i]);

                        dvariances[i] += errors[sample][i] * (log.layerInputs[layerIndex][sample][i] - means[i]);
                        dmeans[i] += errors[sample][i];
                    }

                    dvariances[i] *= (-0.5f) * gamma * MathF.Pow(variances[i], -1.5f);
                    dvariances[i] += 0.000001f;

                    dmeans[i] *= (gamma * sampleSize) / (MathF.Sqrt(variances[i]) * dvariances[i] * 2);
                    // dmeans[i] = (-gamma) / MathF.Sqrt(variances[i]); 
                    // dmeans[i] /= dvariances[i] * (-2) * (1 / sampleSize); 

                    for (int sample = 0; sample < sampleSize; sample++)
                        dmeans[i] += log.layerInputs[layerIndex][sample][i] - means[i];
                    dmeans[i] *= dvariances[i] * (-2);
                    dmeans[i] /= sampleSize;

                    for (int sample = 0; sample < sampleSize; sample++)
                        errors[sample][i] =
                            (errors[sample][i] * gamma) / MathF.Sqrt(variances[i]) +
                            dmeans[i] / sampleSize +
                            (2 * dvariances[i] * (log.layerInputs[layerIndex][sample][i] - means[i])) / sampleSize;
                }

                for (int i = 0; i < dim; i++)
                {
                    gamma = ((IBatchNormOptimizable)optimizer).GammaUpdate(layerIndex, dgamma[i]);
                    beta = ((IBatchNormOptimizable)optimizer).BetaUpdate(layerIndex, dbeta[i]);
                }
            }

            public static float Standardize(float x, float mean, float variance, float zeroSub = 0.000001f) => variance != 0 ? (x - mean) / MathF.Sqrt(variance) : (x - mean) / zeroSub;

        }

        public class NormalizationLayer : ForwardLayer
        {
            public float gamma, beta;

            public NormalizationLayer(float min, float max, ForwardPort port) : base(ActivationFunc.Custom, port, false)
            {
                gamma = 1 / (max - min);
                beta = -min;
            }

            public override void GradientDescent(ref float[][] errors, ForwardResult result, Optimizer optimizer) { }

            public override float[] Forward(float[] X)
            {
                float[] rs = new float[X.Length];
                for (int i = 0; i < X.Length; i++)
                    rs[i] = (X[i] + GetBias(i)) * gamma + beta;
                return rs;
            }
        }

        public class ForwardLayer : ActivationLayer
        {
            public enum ForwardPort
            {
                In,
                Out,
                Both
            }

            public readonly ForwardPort port;

            public ForwardLayer(ActivationFunc func, ForwardPort port, bool useBias = true) : base(-1, func, useBias)
            {
                this.port = port;
            }

            public override void Build(INeuralNetwork network)
            {
                base.Build(network);

                switch (port)
                {
                    case ForwardPort.In:
                        dim = network.Layers[layerIndex - 1].dim;
                        break;
                    case ForwardPort.Out:
                        dim = network.Layers[layerIndex + 1].dim;
                        break;
                    case ForwardPort.Both:
                        if (network.Layers[layerIndex - 1].dim != network.Layers[layerIndex + 1].dim)
                            throw new Exception("Nah forward layer dim");
                        dim = network.Layers[layerIndex + 1].dim;
                        break;
                }

                biases = new float[dim];
            }

            public override WeightMatrix GenerateWeightMatrix()
            {
                return new ForwardWeightMatrix(useBias);
            }
        }

        public class ActivationLayer : Layer
        {
            public readonly ActivationFunc func;

            public ActivationLayer(int dim, ActivationFunc func, bool useBias = true) : base(dim, useBias)
            {
                this.func = func;
            }

            public override float[] Forward(float[] x)
            {
                return ForwardActivation(func, x);
            }

            public override float[] FunctionDifferential(float[] X, float[] loss)
            {
                return ActivationDifferential(func, X, loss);
            }

            public override float FunctionDifferential(float x, float loss, float offset = 0)
            {
                return ActivationDifferential(func, x + offset, loss);
            }

            public static float ActivationDifferential(ActivationFunc func, float x, float loss)
            {
                switch (func)
                {
                    case ActivationFunc.Sigmoid:
                        float sigmoid = ForwardActivation(func, x);
                        return sigmoid * (1 - sigmoid) * loss;
                    case ActivationFunc.Tanh:
                        float sqrExp = MathF.Exp(x) * loss;
                        sqrExp *= sqrExp;
                        return 4 * loss / (sqrExp + (1 / sqrExp) + 2);
                    case ActivationFunc.ReLU:
                        return (x > 0) ? loss : 0;
                    case ActivationFunc.NaturalLog:
                        return loss / x;
                    case ActivationFunc.Exponential:
                        return MathF.Exp(x) * loss;
                    case ActivationFunc.Softmax:
                        return 0;
                    case ActivationFunc.Linear:
                    default:
                        return loss;
                }
            }

            public static float[] ActivationDifferential(ActivationFunc func, float[] X, float[] loss)
            {
                float[] result = new float[X.Length];
                switch (func)
                {
                    case ActivationFunc.Sigmoid:
                        for (int i = 0; i < X.Length; i++)
                        {
                            float sigmoid = ForwardActivation(func, X[i]);
                            result[i] = sigmoid * (1 - sigmoid) * loss[i];
                        }
                        break;
                    case ActivationFunc.Tanh:
                        for (int i = 0; i < X.Length; i++)
                        {
                            float sqrExp = MathF.Exp(X[i]);
                            sqrExp *= sqrExp;
                            result[i] = 4 * loss[i] / (sqrExp + (1 / sqrExp) + 2);
                        }
                        break;
                    case ActivationFunc.ReLU:
                        for (int i = 0; i < X.Length; i++)
                            result[i] = (X[i] > 0) ? loss[i] : 0;
                        break;
                    case ActivationFunc.NaturalLog:
                        for (int i = 0; i < X.Length; i++)
                            result[i] = loss[i] / X[i];
                        break;
                    case ActivationFunc.Exponential:
                        for (int i = 0; i < X.Length; i++)
                            result[i] = MathF.Exp(X[i]) * loss[i];
                        break;
                    case ActivationFunc.Softmax:
                        float[] softmax = ForwardActivation(func, X);
                        for (int i = 0; i < X.Length; i++)
                            for (int j = 0; j < X.Length; j++)
                                result[i] += (i == j) ? softmax[i] * (1 - softmax[i]) * loss[j] : -softmax[i] * softmax[j] * loss[j];
                        break;
                    case ActivationFunc.Linear:
                    default:
                        for (int i = 0; i < X.Length; i++)
                            result[i] = loss[i];
                        break;
                }

                return result;
            }

            public static float ForwardActivation(ActivationFunc func, float x)
            {
                switch (func)
                {
                    case ActivationFunc.Sigmoid:
                        return 1 / (1 + MathF.Exp(-x));
                    case ActivationFunc.Tanh:
                        return MathF.Tanh(x);
                    case ActivationFunc.ReLU:
                        return (x > 0) ? x : 0;
                    case ActivationFunc.NaturalLog:
                        return MathF.Log(x);
                    case ActivationFunc.Exponential:
                        return MathF.Exp(x);
                    case ActivationFunc.Softmax:
                        return 1;
                    case ActivationFunc.Linear:
                    default:
                        return x;
                }
            }

            public static float[] ForwardActivation(ActivationFunc func, float[] X)
            {
                float[] result = new float[X.Length];
                switch (func)
                {
                    case ActivationFunc.Sigmoid:
                        for (int i = 0; i < X.Length; i++)
                            result[i] = 1 / (1 + MathF.Exp(-X[i]));
                        break;
                    case ActivationFunc.Tanh:
                        for (int i = 0; i < X.Length; i++)
                            result[i] = MathF.Tanh(X[i]);
                        break;
                    case ActivationFunc.ReLU:
                        for (int i = 0; i < X.Length; i++)
                            result[i] = (X[i] > 0) ? X[i] : 0;
                        break;
                    case ActivationFunc.NaturalLog:
                        for (int i = 0; i < X.Length; i++)
                            result[i] = MathF.Log(X[i]);
                        break;
                    case ActivationFunc.Exponential:
                        for (int i = 0; i < X.Length; i++)
                            result[i] = MathF.Exp(X[i]);
                        break;
                    case ActivationFunc.Softmax:
                        float temp = 0;
                        for (int i = 0; i < X.Length; i++)
                        {
                            result[i] = MathF.Exp(X[i]);
                            temp += result[i];
                        }
                        temp = 1f / temp;
                        for (int i = 0; i < X.Length; i++)
                            result[i] *= temp;
                        break;
                    case ActivationFunc.Linear:
                    default:
                        return X;
                }

                return result;
            }
        }

        public class Layer
        {
            public readonly bool useBias;

            public int dim { get; protected set; } = -1;
            public INeuralNetwork network { get; protected set; }

            protected int layerIndex = -1;
            protected float[] biases;

            public Layer(int dim, bool useBias = true)
            {
                this.dim = dim;
                this.useBias = useBias;

                if (dim != -1)
                    biases = new float[dim];
            }

            public Layer(float[] biases)
            {
                this.dim = biases.Length;
                this.biases = biases;
            }

            public virtual void Build(INeuralNetwork network)
            {
                this.network = network;
                for (int i = 0; i < network.Layers.Length; i++)
                    if (network.Layers[i] == this)
                    {
                        layerIndex = i;
                        return;
                    }

                throw new Exception("nah layer findings");
            }

            public virtual float GetBias(int index) => useBias ? biases[index] : 0;

            public virtual void SetBias(int index, float value) => biases[index] = useBias ? value : 0;

            /// <returns>Returns descended errors</returns>
            public virtual void GradientDescent(ref float[][] errors, ForwardResult log, Optimizer optimizer)
            {
                if (!useBias)
                    return;

                for (int sample = 0; sample < errors.Length; sample++)
                {
                    errors[sample] = FunctionDifferential(log.layerInputs[layerIndex][sample], errors[sample]);

                    // bias update
                    //for (int i = 0; i < dim; i++)
                    //    SetBias(i, optimizer.BiasUpdate(layerIndex, i, errors[sample][i]));
                }
            }

            public virtual float[][] Forward(float[][] inputs)
            {
                float[][] result = new float[inputs.Length][];

                for (int i = 0; i < inputs.Length; i++)
                    result[i] = new float[dim];

                for (int i = 0; i < inputs.Length; i++)
                    result[i] = Forward(result[i]);

                return result;
            }

            public virtual float[] Forward(float[] inputs)
            {
                float[] result = new float[dim];

                for (int i = 0; i < dim; i++)
                    result[i] = inputs[i] + GetBias(i);

                return result;
            }

            /// <summary>
            /// Will be called indirectly through <b>FunctionDifferential(float[] X)</b> if wasn't overridden
            /// </summary>
            /// <returns> Return <b>df(bias, x) / dx</b></returns>
            public virtual float FunctionDifferential(float x, float loss, float offset = 0f) => loss;

            /// <summary>
            /// Will be called directly in the <b>GradientDescent</b> method
            /// </summary>
            /// <returns> Return <b>df(bias, x) / dx</b> for each x in X </returns>
            public virtual float[] FunctionDifferential(float[] X, float[] loss)
            {
                float[] result = new float[X.Length];
                for (int i = 0; i < X.Length; i++)
                    result[i] = loss[i];

                return result;
            }

            public virtual WeightMatrix GenerateWeightMatrix()
            {
                return new DenseWeightMatrix();
            }

            public static implicit operator Layer(int dim) => new Layer(dim);
        }

        #endregion

        #region Weight matrix

        public abstract class WeightMatrix
        {
            public int inDim { get; protected set; }
            public int outDim { get; protected set; }
            public INeuralNetwork network { get; protected set; }

            protected int weightsIndex;

            public virtual void Build(INeuralNetwork network)
            {
                this.network = network;
                for (int i = 0; i < network.Weights.Length; i++)
                    if (network.Weights[i] == this)
                    {
                        weightsIndex = i;
                        return;
                    }

                throw new Exception("nah weights findings");
            }

            public abstract void GradientDescent(ref float[][] errors, ForwardResult result, Optimizer optimizer);

            public abstract float[] Forward(float[] inputs);

            public abstract float[][] Forward(float[][] inputs);

            public abstract float ForwardComp(float[] inputs, int outputIndex);

            /// <summary>
            /// 
            /// </summary>
            /// <param name="value"><inIndex, outIndex, weightValue, returnValue></param>
            public abstract void AssignForEach(Func<int, int, float, float> value);

            public abstract bool TrySetWeight(int inIndex, int outIndex, float value);

            public abstract bool TryGetWeight(int inIndex, int outIndex, out float weight);

            public abstract float GetWeight(int inIndex, int outIndex);
        }

        public class ForwardWeightMatrix : WeightMatrix
        {
            public readonly bool useWeights;

            public float[] matrix;

            public int dim => inDim;

            public ForwardWeightMatrix(bool useWeights = true)
            {
                this.useWeights = useWeights;
            }

            public override void Build(INeuralNetwork network)
            {
                base.Build(network);

                if (network.Layers[weightsIndex] is ForwardLayer && ((ForwardLayer)network.Layers[weightsIndex]).port != ForwardLayer.ForwardPort.In)
                    inDim = outDim = network.Layers[weightsIndex].dim;
                else if (network.Layers[weightsIndex + 1] is ForwardLayer && ((ForwardLayer)network.Layers[weightsIndex + 1]).port != ForwardLayer.ForwardPort.Out)
                    inDim = outDim = network.Layers[weightsIndex + 1].dim;
                else
                    throw new Exception("Nah forward weight dim");

                matrix = new float[dim];
            }

            public override void AssignForEach(Func<int, int, float, float> value)
            {
                for (int i = 0; i < dim; i++)
                {
                    if (useWeights)
                        matrix[i] = value(i, i, matrix[i]);
                    else
                        value(i, i, 1);
                }
            }

            public override void GradientDescent(ref float[][] errors, ForwardResult log, Optimizer optimizer)
            {
                if (!useWeights) return;

                float[] weightErrorSum = new float[matrix.Length];
                for (int sample = 0; sample < errors.Length; sample++)
                {
                    float[] layerForward = network.Layers[weightsIndex].Forward(log.layerInputs[weightsIndex][sample]);
                    float[] layerDif = network.Layers[weightsIndex].FunctionDifferential(log.layerInputs[weightsIndex][sample], errors[sample]);

                    for (int i = 0; i < matrix.Length; i++)
                    {
                        weightErrorSum[i] += errors[sample][i] * layerForward[i];
                        errors[sample][i] = matrix[i] * layerDif[i];
                    }
                }

                for (int i = 0; i < matrix.Length; i++)
                    matrix[i] = optimizer.WeightUpdate(weightsIndex, i, i, weightErrorSum[i]);
            }

            public override float[] Forward(float[] inputs)
            {
                float[] result = new float[dim];

                for (int i = 0; i < dim; i++)
                    result[i] = ForwardComp(inputs, i);

                return result;
            }

            public override float[][] Forward(float[][] inputs)
            {
                float[][] result = new float[inputs.Length][];

                for (int i = 0; i < inputs.Length; i++)
                    result[i] = new float[dim];

                for (int i = 0; i < inputs.Length; i++)
                    for (int j = 0; j < dim; j++)
                    {
                        if (useWeights)
                            result[i][j] = inputs[i][j] * matrix[j];
                        else
                            result[i][j] = inputs[i][j];
                    }

                return result;
            }

            public override float ForwardComp(float[] inputs, int outputIndex)
            {
                if (useWeights)
                    return inputs[outputIndex] * matrix[outputIndex];
                else
                    return inputs[outputIndex];
            }

            public override float GetWeight(int inIndex, int outIndex)
            {
                if (useWeights)
                {
                    if (inIndex == outIndex && inIndex < dim)
                        return matrix[inIndex];
                }
                else if (inIndex == outIndex)
                    return 1;
                else
                    return 0;

                throw new Exception("No weight here bro");
            }

            public override bool TryGetWeight(int inIndex, int outIndex, out float weight)
            {
                if (useWeights)
                    weight = matrix[inIndex];
                else if (inIndex == outIndex)
                    weight = 1;
                else
                    weight = 0;

                return inIndex == outIndex && inIndex < dim;
            }

            public override bool TrySetWeight(int inIndex, int outIndex, float value)
            {
                if (useWeights && inIndex == outIndex && inIndex < dim)
                {
                    matrix[inIndex] = value;
                    return true;
                }

                return false;
            }
        }

        public class DenseWeightMatrix : WeightMatrix
        {
            public float[,] matrix;

            public DenseWeightMatrix() { }

            public override void Build(INeuralNetwork network)
            {
                base.Build(network);

                inDim = network.Layers[weightsIndex].dim;
                outDim = network.Layers[weightsIndex + 1].dim;

                matrix = new float[outDim, inDim];
            }

            public override void GradientDescent(ref float[][] errors, ForwardResult log, Optimizer optimizer)
            {
                Layer prevLayer = network.Layers[weightsIndex];

                float[][] weightErrors = new float[errors.Length][];
                for (int i = 0; i < errors.Length; i++)
                    weightErrors[i] = new float[inDim];

                float[][] weightErrorSum = new float[outDim][];
                for (int i = 0; i < outDim; i++)
                    weightErrorSum[i] = new float[inDim];

                for (int sample = 0; sample < errors.Length; sample++)
                {
                    float[] layerForward = prevLayer.Forward(log.layerInputs[weightsIndex][sample]);

                    for (int i = 0; i < outDim; i++)
                        for (int j = 0; j < inDim; j++)
                        {
                            weightErrorSum[i][j] += errors[sample][i] * layerForward[j];
                            weightErrors[sample][j] += errors[sample][i] * matrix[i, j];
                        }

                    weightErrors[sample] = prevLayer.FunctionDifferential(log.layerInputs[weightsIndex][sample], weightErrors[sample]);
                }

                for (int i = 0; i < outDim; i++)
                    for (int j = 0; j < inDim; j++)
                        matrix[i, j] = optimizer.WeightUpdate(weightsIndex, j, i, weightErrorSum[i][j]);

                errors = weightErrors;
            }

            public override bool TryGetWeight(int inIndex, int outIndex, out float weight)
            {
                weight = matrix[outIndex, inIndex];
                return true;
            }

            public override float GetWeight(int inIndex, int outIndex) => matrix[outIndex, inIndex];

            public override void AssignForEach(Func<int, int, float, float> value)
            {
                for (int i = 0; i < outDim; i++)
                    for (int j = 0; j < inDim; j++)
                        matrix[i, j] = value(j, i, matrix[i, j]);
            }

            public override float[] Forward(float[] inputs)
            {
                float[] result = new float[outDim];

                for (int i = 0; i < outDim; i++)
                    for (int j = 0; j < inDim; j++)
                        result[i] += inputs[j] * matrix[i, j];

                return result;
            }

            public override float[][] Forward(float[][] inputs)
            {
                float[][] result = new float[inputs.Length][];

                for (int i = 0; i < inputs.Length; i++)
                    result[i] = new float[outDim];

                for (int i = 0; i < inputs.Length; i++)
                    for (int j = 0; j < outDim; j++)
                        for (int k = 0; k < inDim; k++)
                            result[i][j] += inputs[i][k] * matrix[j, k];

                return result;
            }

            public override float ForwardComp(float[] inputs, int outputIndex)
            {
                float output = 0;

                for (int i = 0; i < inputs.LongLength; i++)
                    output += matrix[outputIndex, i] * inputs[i];

                return output;
            }

            public override bool TrySetWeight(int inIndex, int outIndex, float value)
            {
                matrix[outIndex, inIndex] = value;
                return true;
            }
        }

        #endregion

        namespace RL
        {
            public enum ConcludeType
            {
                None,
                Terminate,
                Killed,
            }

            public interface IEnvironment
            {
                IEnumerable<IAgent> Agents { get; }

                void Init();

                float Evaluate(IAgent agent);

                /// <summary>
                /// Reset all including environment and agents
                /// </summary>
                void ResetStates();
            }

            public interface IAgent
            {
                IEnvironment Env { get; }
                IPolicy Policy { get; }
                IPolicyOptimization PolicyOpt { get; }

                /// <summary>
                /// Reset states, not policy
                /// </summary>
                void ResetStates();

                /// <returns><b>Observation</b> as float[] and <b>reward</b> as float</returns>
                void TakeAction();

                void Conclude(ConcludeType type);

            }

            public interface IPolicy
            {
                float[] Forward(float[] obs);

                void Update(float[] loss);
            }

            public interface IPolicyOptimization
            {
                IPolicy Policy { get; }

                int GetAction(float[] actProbs);

                virtual float[][] ComputeLoss(float[][] obs, int[] actions, float[] mass, bool logits = false)
                {
                    float[][] loss = new float[obs.Length][];
                    for (int i = 0; i < loss.Length; i++)
                        loss[i] = ComputeLoss(obs[i], actions[i], mass[i], logits);

                    return loss;
                }

                float[] ComputeLoss(float[] obs, int action, float mass, bool logits = false);
            }

            public class Reinforce : IPolicyOptimization
            {
                public IPolicy Policy { get; private set; }

                public Reinforce(IPolicy policy)
                {
                    Policy = policy;
                }

                public virtual int GetAction(float[] actProbs) => MathBT.DrawProbs(actProbs);

                public virtual float[] ComputeLoss(float[] obs, int action, float mass, bool logits = false)
                {
                    float[] outputs;
                    if (logits)
                        outputs = Policy.Forward(obs);
                    else
                        outputs = ActivationLayer.ForwardActivation(ActivationFunc.Softmax, Policy.Forward(obs));

                    for (int i = 0; i < outputs.Length; i++)
                    {
                        outputs[i] = i == action ? -(1 / outputs[i]) * mass : 0;
                    }

                    return outputs;
                }
            }

            public class ExplorationWrapper : IPolicyOptimization
            {
                public IPolicyOptimization content;
                public float exploreRate, exploreDecay;

                public IPolicy Policy => content.Policy;

                public ExplorationWrapper(IPolicyOptimization content, float initialRate, float decay)
                {
                    this.content = content;
                    exploreRate = initialRate;
                    exploreDecay = decay;
                }

                public int GetAction(float[] actProbs)
                {
                    int act = -1;
                    Random rand = new();
                    if (rand.NextDouble() > exploreRate)
                        act = content.GetAction(actProbs);
                    else
                        act = rand.Next(actProbs.Length);
                    exploreRate *= exploreDecay;
                    return act;
                }

                public float[] ComputeLoss(float[] obs, int action, float mass, bool logits = false)
                    => content.ComputeLoss(obs, action, mass, logits);
            }
        }
    }

    namespace Data
    {
        public static class UData
        {
            public static string[] GetCategoriesFromCSV(string path)
            {
                string[] cats;
                using (StreamReader reader = new StreamReader(path))
                    cats = reader.ReadLine().Split(',');

                return cats;
            }

            public static Dictionary<string, float>[] RetrieveDistinctIntDataFromCSV(string path, int retrieveAmount, params string[] retrieveCats)
            {
                string[] cats = GetCategoriesFromCSV(path);
                List<string> neglectCats = new List<string>();
                DistinctIntDataInfo[] encodings;
                Dictionary<string, AdditionNumericDataInfo> numericInfos;

                DataType[] dataTypes = new DataType[cats.Length];

                for (int i = 0; i < cats.Length; i++)
                {
                    bool going = true;
                    foreach (string cat in retrieveCats)
                        if (cats[i] == cat)
                        {
                            dataTypes[i] = DataType.DistinctInt;
                            going = false;
                            break;
                        }

                    if (!going)
                        continue;

                    dataTypes[i] = DataType.Neglect;
                }

                for (int i = 0; i < dataTypes.Length; i++)
                    if (dataTypes[i] == DataType.Neglect)
                        neglectCats.Add(cats[i]);

                UDataInfo info = new UDataInfo(neglectCats.ToArray(), dataTypes);

                return RetrieveUDataFromCSV(path, info, out cats, out encodings, out numericInfos, retrieveAmount);
            }

            public static Dictionary<string, float>[] RetrieveNumberDataFromCSV(string path, int retrieveAmount, out Dictionary<string, AdditionNumericDataInfo> numericInfos, params string[] retrieveCats)
            {
                string[] cats = GetCategoriesFromCSV(path);
                List<string> neglectCats = new List<string>();
                DistinctIntDataInfo[] encodings;

                DataType[] dataTypes = new DataType[cats.Length];

                for (int i = 0; i < cats.Length; i++)
                {
                    bool going = true;
                    foreach (string cat in retrieveCats)
                        if (cats[i] == cat)
                        {
                            dataTypes[i] = DataType.Float;
                            going = false;
                            break;
                        }

                    if (!going)
                        continue;

                    dataTypes[i] = DataType.Neglect;
                }

                for (int i = 0; i < dataTypes.Length; i++)
                    if (dataTypes[i] == DataType.Neglect)
                        neglectCats.Add(cats[i]);

                UDataInfo info = new UDataInfo(neglectCats.ToArray(), dataTypes);

                return RetrieveUDataFromCSV(path, info, out cats, out encodings, out numericInfos, retrieveAmount);
            }

            public static Dictionary<string, float>[] RetrieveUDataFromCSV(string path, UDataInfo info, int retrieveAmount = -1)
            {
                string[] cats;
                DistinctIntDataInfo[] encodings;
                Dictionary<string, AdditionNumericDataInfo> numericInfos;
                return RetrieveUDataFromCSV(path, info, out cats, out encodings, out numericInfos, retrieveAmount);
            }

            public static Dictionary<string, float>[] RetrieveUDataFromCSV(string path, UDataInfo info, out Dictionary<string, AdditionNumericDataInfo> numericInfos, int retrieveAmount = -1)
            {
                string[] cats;
                DistinctIntDataInfo[] encodings;
                return RetrieveUDataFromCSV(path, info, out cats, out encodings, out numericInfos, retrieveAmount);
            }

            public static Dictionary<string, float>[] RetrieveUDataFromCSV(string path, UDataInfo info, out DistinctIntDataInfo[] distinctEncodings, out Dictionary<string, AdditionNumericDataInfo> numericInfos, int retrieveAmount = -1)
            {
                string[] cats;
                return RetrieveUDataFromCSV(path, info, out cats, out distinctEncodings, out numericInfos, retrieveAmount);
            }

            public static Dictionary<string, float>[] RetrieveUDataFromCSV(string path, UDataInfo info, out string[] categories, out DistinctIntDataInfo[] distinctEncodings, out Dictionary<string, AdditionNumericDataInfo> numericInfos, int retrieveAmount = -1)
            {
                List<Dictionary<string, float>> data = new List<Dictionary<string, float>>();
                List<float[]> rawData = new List<float[]>();
                numericInfos = new Dictionary<string, AdditionNumericDataInfo>();

                using (StreamReader reader = new StreamReader(path))
                {
                    categories = reader.ReadLine().Split(',');

                    if (info.types.Length != categories.Length)
                        throw new Exception("type info unmatch");

                    List<int> iteratingIndexList = new List<int>();

                    for (int i = 0; i < categories.Length; i++)
                    {
                        int j = 0;
                        for (; j < info.neglectCats.Length; j++)
                            if (categories[i] == info.neglectCats[j])
                                break;

                        if (j == info.neglectCats.Length)
                            iteratingIndexList.Add(i);
                    }

                    int[] iteratingIndices = iteratingIndexList.ToArray();

                    foreach (int i in iteratingIndices)
                    {
                        if (info.types[i] == DataType.Float)
                            numericInfos.Add(categories[i], new AdditionNumericDataInfo(float.NaN, float.NaN, 0));
                    }

                    Dictionary<int, int> givenDistinctDataIndices = new Dictionary<int, int>();

                    for (int i = 0; i < info.distinctData.Length; i++)
                        for (int j = 0; j < categories.Length; j++)
                            if (info.distinctData[i].category == categories[j])
                            {
                                givenDistinctDataIndices.Add(j, i);
                                break;
                            }

                    List<int> scoutDistinctDataIndices = new List<int>();
                    for (int i = 0; i < info.types.Length; i++)
                        if (info.types[i] == DataType.DistinctInt)
                            if (!givenDistinctDataIndices.Keys.Contains(i))
                                scoutDistinctDataIndices.Add(i);

                    List<string>[] scoutDistinctData = new List<string>[scoutDistinctDataIndices.Count];

                    for (int i = 0; i < scoutDistinctData.Length; i++)
                        scoutDistinctData[i] = new List<string>();

                    int retrieveCount = 0;
                    while (!reader.EndOfStream && (retrieveCount < retrieveAmount || retrieveAmount == -1))
                    {
                        string[] rawDataLine = reader.ReadLine().Split(',');
                        float[] dataLine = new float[iteratingIndices.Length];

                        int curDistinct = 0;

                        bool empty = false;
                        foreach (int index in iteratingIndices)
                            if (string.IsNullOrEmpty(rawDataLine[index]))
                            {
                                empty = true;
                                break;
                            }
                        if (empty) continue;

                        for (int i = 0; i < iteratingIndices.Length; i++)
                        {
                            if (string.IsNullOrEmpty(rawDataLine[iteratingIndices[i]]))
                                continue;

                            bool added = false;
                            foreach (int index in givenDistinctDataIndices.Keys)
                                if (iteratingIndices[i] == index)
                                {
                                    for (int j = 0; j < info.distinctData[givenDistinctDataIndices[index]].encodings.Length; j++)
                                        if (info.distinctData[givenDistinctDataIndices[index]].encodings[j] == rawDataLine[iteratingIndices[i]])
                                            dataLine[iteratingIndices[i]] = j;

                                    added = true;
                                }

                            if (added)
                                continue;

                            switch (info.types[iteratingIndices[i]])
                            {
                                case DataType.Float:
                                    dataLine[i] = float.Parse(rawDataLine[iteratingIndices[i]]);

                                    float min = numericInfos[categories[iteratingIndices[i]]].min;
                                    float max = numericInfos[categories[iteratingIndices[i]]].max;

                                    if (min > dataLine[i] || float.IsNaN(min)) min = dataLine[i];
                                    if (max < dataLine[i] || float.IsNaN(max)) max = dataLine[i];

                                    numericInfos[categories[iteratingIndices[i]]] = new AdditionNumericDataInfo(min, max, numericInfos[categories[iteratingIndices[i]]].mean + dataLine[i]);

                                    break;
                                case DataType.DistinctInt:
                                    int j = 0;
                                    for (; j < scoutDistinctData[curDistinct].Count;)
                                    {
                                        if (scoutDistinctData[curDistinct][j] == rawDataLine[iteratingIndices[i]])
                                        {
                                            dataLine[i] = j;
                                            break;
                                        }
                                        j++;
                                    }

                                    if (j != scoutDistinctData[curDistinct].Count)
                                    {
                                        curDistinct++;
                                        break;
                                    }

                                    scoutDistinctData[curDistinct].Add(rawDataLine[iteratingIndices[i]]);
                                    dataLine[i] = scoutDistinctData[curDistinct].Count - 1;
                                    curDistinct++;
                                    break;
                            }
                        }

                        rawData.Add(dataLine);
                        retrieveCount++;
                    }

                    DistinctIntDataInfo[] distinctInfos = new DistinctIntDataInfo[scoutDistinctData.Length + info.distinctData.Length];

                    foreach (int i in iteratingIndices)
                        if (info.types[i] == DataType.Float)
                            numericInfos[categories[i]] = new AdditionNumericDataInfo(numericInfos[categories[i]].min, numericInfos[categories[i]].max, numericInfos[categories[i]].mean / rawData.Count);

                    for (int i = 0; i < scoutDistinctData.Length; i++)
                        distinctInfos[i] = new DistinctIntDataInfo(categories[scoutDistinctDataIndices[i]], scoutDistinctData[i].ToArray());

                    for (int i = scoutDistinctData.Length; i < info.distinctData.Length; i++)
                        distinctInfos[i] = info.distinctData[i - scoutDistinctData.Length];

                    distinctEncodings = distinctInfos;

                    for (int sample = 0; sample < rawData.Count; sample++)
                    {
                        data.Add(new Dictionary<string, float>());

                        for (int i = 0; i < iteratingIndices.Length; i++)
                            data[sample].Add(categories[iteratingIndices[i]], rawData[sample][i]);
                    }
                }

                return data.ToArray();
            }
        }

        public enum NormalizationMode
        {
            MinMaxRange,
            DivideMean,
            CutMinDivideMean
        }

        public struct AdditionNumericDataInfo
        {
            public float min, max, mean;

            public AdditionNumericDataInfo(float min, float max, float mean)
            {
                this.min = min;
                this.max = max;
                this.mean = mean;
            }

            public float Normalize(NormalizationMode mode, float value)
            {
                switch (mode)
                {
                    case NormalizationMode.MinMaxRange:
                        return (value - min) / (max - min);
                    case NormalizationMode.DivideMean:
                        return value / mean;
                    case NormalizationMode.CutMinDivideMean:
                        return (value - min) / (mean - min);
                    default:
                        return value;
                }
            }

            public float Denormalize(NormalizationMode mode, float value)
            {
                switch (mode)
                {
                    case NormalizationMode.MinMaxRange:
                        return value * (max - min) + min;
                    case NormalizationMode.DivideMean:
                        return value * mean;
                    case NormalizationMode.CutMinDivideMean:
                        return (value - 1) * (mean - min) + min;
                    default:
                        return value;
                }
            }
        }

        public enum DataType
        {
            Neglect,
            Float,
            DistinctInt
        }

        public struct UDataInfo
        {
            public DataType[] types;
            public DistinctIntDataInfo[] distinctData;
            public string[] neglectCats;

            public UDataInfo(string[] categories, params Tuple<string, DataType>[] catTypes)
            {
                types = new DataType[categories.Length];
                for (int i = 0; i < categories.Length; i++)
                {
                    types[i] = DataType.Neglect;
                    foreach (var catType in catTypes)
                        if (catType.Item1 == categories[i])
                        {
                            types[i] = catType.Item2;
                            break;
                        }
                }

                List<string> negCatList = new List<string>();
                for (int i = 0; i < categories.Length; i++)
                    if (types[i] == DataType.Neglect)
                        negCatList.Add(categories[i]);

                this.distinctData = new DistinctIntDataInfo[0];
                this.neglectCats = negCatList.ToArray();
            }

            public UDataInfo(string[] neglectCats, params DataType[] types)
            {
                this.types = types;
                this.distinctData = new DistinctIntDataInfo[0];
                this.neglectCats = neglectCats;
            }

            public UDataInfo(DistinctIntDataInfo[] distinctData, string[] neglectCats, params DataType[] types)
            {
                this.types = types;
                this.distinctData = distinctData;
                this.neglectCats = neglectCats;
            }

            public UDataInfo(DistinctIntDataInfo[] distinctData, params DataType[] types)
            {
                this.types = types;
                this.distinctData = distinctData;
                this.neglectCats = new string[0];
            }

            public UDataInfo(params DataType[] types)
            {
                this.types = types;
                this.distinctData = new DistinctIntDataInfo[0];
                this.neglectCats = new string[0];
            }
        }

        public struct DistinctIntDataInfo
        {
            public string category;
            public string[] encodings;

            public DistinctIntDataInfo(string category, string[] encodings)
            {
                this.category = category;
                this.encodings = encodings;
            }
        }
    }

    namespace Utility
    {
        public static class MathBT
        {
            public static int DrawProbs(float[] probs)
            {
                double rand = new Random().NextDouble();

                for (int i = 0; i < probs.Length; i++)
                {
                    rand -= probs[i];
                    if (rand <= 0)
                        return i;
                }

                return -1;
            }
        }

        namespace Linear
        {
            public static class LinearMethod
            {
                public static Vector CGMethod(ISquareMatrix A, Vector b, float epsilon = 1E-6f)
                {
                    if (A.dim != b.dim)
                        throw new Exception("Invalid Conjugate Gradient Method input dims");

                    Vector
                        result = new float[A.dim],
                        residual = Vector.Clone(b.content), preResidual = Vector.Clone(b.content),
                        direction = Vector.Clone(residual.content), scaledDirection = A * direction;

                    if (Vector.Dot(residual, residual) > epsilon)
                    {
                        float alpha = Vector.Dot(residual, residual) / Vector.Dot(direction, scaledDirection);

                        result += alpha * direction;
                        residual -= alpha * scaledDirection;
                    }
                    while (Vector.Dot(residual, residual) > epsilon)
                    {
                        float beta = Vector.Dot(residual, residual) / Vector.Dot(preResidual, preResidual);

                        direction = residual + beta * direction;
                        scaledDirection = A * direction;

                        float alpha = Vector.Dot(residual, residual) / Vector.Dot(direction, scaledDirection);

                        preResidual.SetTo(residual);
                        result += alpha * direction;
                        residual -= alpha * scaledDirection;
                    }

                    return result.content;
                }

                public static Vector CGMethod(ISquareMatrix A, Vector b, IPreconditioner preconditioner, float epsilon = 1E-6f)
                {
                    if (A.dim != b.dim)
                        throw new Exception("Invalid Conjugate Gradient Method input dims");

                    preconditioner.Init(A, b);

                    Vector
                        result = new float[A.dim],
                        residual = Vector.Clone(b.content), preResidual = Vector.Clone(b.content),
                        preconditionedResidual = preconditioner.Forward(residual), prePreconditionedResidual = preconditioner.Forward(residual),
                        direction = Vector.Clone(preconditionedResidual.content), scaledDirection = A * direction;

                    if (Vector.Dot(residual, residual) > epsilon)
                    {
                        float alpha = Vector.Dot(residual, preconditionedResidual) / Vector.Dot(direction, scaledDirection);

                        result += alpha * direction;
                        residual -= alpha * scaledDirection;
                    }

                    while (Vector.Dot(residual, residual) > epsilon)
                    {
                        preconditionedResidual = preconditioner.Forward(residual);

                        float beta = Vector.Dot(residual, preconditionedResidual) / Vector.Dot(preResidual, prePreconditionedResidual);

                        direction = preconditionedResidual + beta * direction;
                        scaledDirection = A * direction;

                        float alpha = Vector.Dot(residual, preconditionedResidual) / Vector.Dot(direction, scaledDirection);

                        preResidual.SetTo(residual);
                        prePreconditionedResidual.SetTo(preconditionedResidual);
                        result += alpha * direction;
                        residual -= alpha * scaledDirection;
                    }

                    return result.content;
                }

                public static Vector CGNEMethod(IMatrix A, Vector b, float epsilon = 1E-6f)
                {
                    if (A.rowCount != b.dim)
                        throw new Exception("Invalid Conjugate Gradient Method input dims");
                    IMatrix At = A.Transpose;

                    return CGMethod((At * A).ToSquare, At * b, epsilon);
                }

                public static Vector CGNEMethod(IMatrix A, Vector b, IPreconditioner preconditioner, float epsilon = 1E-6f)
                {
                    if (A.rowCount != b.dim)
                        throw new Exception("Invalid Conjugate Gradient Method input dims");
                    IMatrix At = A.Transpose;

                    return CGMethod((At * A).ToSquare, At * b, preconditioner, epsilon);
                }

                /// <returns>A tuple of a lower matrix and an upper LU factorizations respectively</returns>
                public static (TriangularMatrix, TriangularMatrix) IncompleteLUFac(ISquareMatrix A, float epsilon = 1E-3f)
                {
                    TriangularMatrix lower = new TriangularMatrix(A.dim, false);
                    TriangularMatrix upper = new TriangularMatrix(A.dim, true);

                    for (int i = 0; i < A.dim; i++)
                        for (int j = 0; j <= i; j++)
                        {
                            if (A.Get(j, i) > epsilon)
                            {
                                // Row iterate
                                // j : row index
                                float rowSum = 0;
                                for (int k = 0; k < j; k++)
                                    rowSum += lower.Get(j, k) * upper.Get(k, i);
                                upper.Set(j, i, A.Get(j, i) - rowSum);
                            }

                            if (A.Get(i, j) > epsilon)
                            {
                                // Column iterate
                                // j : column index
                                if (i == j)
                                    lower.Set(i, j, 1);
                                else
                                {
                                    float colSum = 0;
                                    for (int k = 0; k < j; k++)
                                        colSum += lower.Get(i, k) * upper.Get(k, j);

                                    lower.Set(i, j, (A.Get(i, j) - colSum) / upper.Get(j, j));
                                }
                            }

                        }

                    return (lower, upper);
                }

                /// <returns>Lower triangular factorization</returns>
                public static TriangularMatrix IncompleteCholeskyFac(ISquareMatrix A, float epsilon = 1E-3f)
                {
                    TriangularMatrix result = new TriangularMatrix(A.dim, false);

                    for (int row = 0; row < A.dim; row++)
                        for (int col = 0; col < row + 1; col++)
                        {
                            if (A.Get(row, col) < epsilon)
                            {
                                result.Set(row, col, 0);
                                continue;
                            }

                            float sum = 0;
                            for (int i = 0; i < col; i++)
                                sum += result.Get(row, i) * result.Get(col, i);

                            if (col == row)
                                result.Set(row, col, MathF.Sqrt(A.Get(row, col) - sum));
                            else
                                result.Set(row, col, (A.Get(row, col) - sum) / result.Get(col, col));
                        }

                    return result;
                }

                public interface IPreconditioner
                {
                    public void Init(ISquareMatrix A, Vector b);

                    public Vector Forward(Vector value);
                }

                public class LUPreconditioner : IPreconditioner
                {
                    public TriangularMatrix lower { get; protected set; }
                    public TriangularMatrix upper { get; protected set; }

                    public virtual void Init(ISquareMatrix A, Vector b)
                    {
                        (lower, upper) = IncompleteLUFac(A);
                    }

                    public virtual Vector Forward(Vector value) => upper.Substitute(lower.Substitute(value));
                }

                public class CholeskyPreconditioner : LUPreconditioner
                {
                    public override void Init(ISquareMatrix A, Vector b)
                    {
                        lower = IncompleteCholeskyFac(A);
                        upper = (TriangularMatrix)lower.Transpose;
                    }
                }
            }

            public interface IMatrix
            {
                public int rowCount { get; }
                public int colCount { get; }

                public IMatrix Transpose { get; }
                public ISquareMatrix ToSquare { get; }

                public IMatrix Instance();
                public IMatrix InstanceT();

                public float Get(int row, int col);
                public bool Set(int row, int col, float value);

                public bool SetTo(IMatrix matrix);

                public IMatrix Clone();

                public static int NegOneRaiseTo(int num)
                {
                    return num % 2 == 0 ? 1 : -1;
                }

                public static IMatrix Identity(int dim)
                {
                    IMatrix identity = new DiagonalMatrix(dim);

                    for (int i = 0; i < dim; i++)
                        identity.Set(i, i, 1);

                    return identity;
                }

                public static IMatrix Diag(params float[] nums)
                {
                    IMatrix matrix = new DiagonalMatrix(nums.Length);

                    for (int i = 0; i < nums.Length; i++)
                        matrix.Set(i, i, nums[i]);

                    return matrix;
                }

                public Vector Multiply(Vector vector);
                public Vector LeftMultiply(Vector vector);

                public IMatrix Multiply(IMatrix matrix);

                public IMatrix Add(float value);

                public IMatrix Subtract(float value);
                public IMatrix LeftSubtract(float value);

                public IMatrix Multiply(float value);

                public IMatrix Divide(float value);

                public static IMatrix operator *(IMatrix A, IMatrix B) => A.Multiply(B);
                public static Vector operator *(IMatrix matrix, Vector vector) => matrix.Multiply(vector);
                public static Vector operator *(Vector vector, IMatrix matrix) => matrix.LeftMultiply(vector);

                public static IMatrix operator +(IMatrix matrix, float value) => matrix.Add(value);
                public static IMatrix operator -(IMatrix matrix, float value) => matrix.Subtract(value);
                public static IMatrix operator *(IMatrix matrix, float value) => matrix.Multiply(value);
                public static IMatrix operator /(IMatrix matrix, float value) => matrix.Divide(value);

                public static IMatrix operator +(float value, IMatrix matrix) => matrix.Add(value);
                public static IMatrix operator -(float value, IMatrix matrix) => matrix.LeftSubtract(value);
                public static IMatrix operator *(float value, IMatrix matrix) => matrix.Multiply(value);
            }

            public interface ISquareMatrix : IMatrix
            {
                public int dim { get; }

                public ISquareMatrix Invert();

                public float Determinant();

                public float Cofactor(int row, int col);

                public ISquareMatrix Adjugate();

                public static float Cofactor(ISquareMatrix matrix, AdjugateSum adj, float multiplier, bool coefMultiply = false)
                {
                    if (multiplier == 0)
                        return 0;

                    if (matrix.dim - adj.Count == 2)
                    {
                        int sum = (matrix.dim - 1) * matrix.dim / 2;
                        int row1 = adj.SmallestAdjugatableRow, row2 = sum - adj.RowSum - row1,
                            col1 = adj.SmallestAdjugatableCol, col2 = sum - adj.ColSum - col1;

                        return multiplier * (matrix.Get(row1, col1) * matrix.Get(row2, col2) - matrix.Get(row1, col2) * matrix.Get(row2, col1));
                    }

                    float result = 0;

                    for (int i = 0; i < matrix.dim; i++)
                    {
                        int rowSkip = 0;
                        var node = adj.rows.First;
                        while (node != null && node.Value < i)
                        {
                            node = node.Next;
                            rowSkip++;
                        }

                        if (node != null && node.Value == i)
                            continue;

                        LinkedListNode<int> rowNode, colNode;

                        int adjCol = adj.SmallestAdjugatableCol, skipCol = adj.SkipAdjCol;
                        adj.Add(i, adjCol, out rowNode, out colNode);
                        result += Cofactor(matrix, adj, NegOneRaiseTo(i - rowSkip + adjCol - skipCol)) * matrix.Get(i, adjCol);
                        adj.Remove(rowNode, colNode);
                    }

                    return result * multiplier;
                }

                public static float Cofactor(ISquareMatrix matrix, int row, int col)
                {
                    if (matrix.dim == 2)
                        return NegOneRaiseTo(row + col) * matrix.Get(1 - row, 1 - col);

                    return Cofactor(matrix, new AdjugateSum(row, col), NegOneRaiseTo(row + col));
                }

                public class AdjugateSum
                {
                    public LinkedList<int> rows, cols;

                    public int SmallestAdjugatableRow { get; private set; } = 0;
                    public int SmallestAdjugatableCol { get; private set; } = 0;
                    public int SkipAdjCol { get; private set; } = 0;
                    public int RowSum { get; private set; } = 0;
                    public int ColSum { get; private set; } = 0;
                    public int Count => rows.Count;

                    public AdjugateSum()
                    {
                        rows = new LinkedList<int>();
                        cols = new LinkedList<int>();
                    }

                    public AdjugateSum(int row, int col)
                    {
                        rows = new LinkedList<int>();
                        cols = new LinkedList<int>();

                        Add(row, col);
                    }

                    public void UpdateColSkip()
                    {
                        SkipAdjCol = 0;
                        var node = cols.First;
                        while (node != null && node.Value < SmallestAdjugatableCol)
                        {
                            node = node.Next;
                            SkipAdjCol++;
                        }
                    }

                    public void Remove(LinkedListNode<int> rowNode, LinkedListNode<int> colNode)
                    {
                        RowSum -= rowNode.Value;
                        ColSum -= colNode.Value;

                        if (rowNode.Value < SmallestAdjugatableRow)
                            SmallestAdjugatableRow = rowNode.Value;

                        if (colNode.Value < SmallestAdjugatableCol)
                            SmallestAdjugatableCol = colNode.Value;

                        rows.Remove(rowNode);
                        cols.Remove(colNode);
                        UpdateColSkip();
                    }

                    public void Add(int row, int col)
                    {
                        LinkedListNode<int> rowNode, colNode;
                        Add(row, col, out rowNode, out colNode);
                    }

                    public void Add(int row, int col, out LinkedListNode<int> rowNode, out LinkedListNode<int> colNode)
                    {
                        LinkedListNode<int> node;
                        rowNode = null;
                        colNode = null;

                        bool added = false;
                        node = rows.First;
                        for (; node != null; node = node.Next)
                        {
                            if (node.Value > row)
                            {
                                RowSum += row;
                                rowNode = rows.AddBefore(node, row);
                                added = true;

                                if (SmallestAdjugatableRow < row)
                                    break;
                                row++;

                                while (node != null && node.Value == row)
                                {
                                    node = node.Next;
                                    row++;
                                }
                                SmallestAdjugatableRow = row;
                                break;

                            }
                            else if (row == node.Value)
                                break;
                        }

                        if (!added)
                        {
                            RowSum += row;
                            rowNode = rows.AddLast(row);

                            if (SmallestAdjugatableRow >= row)
                                SmallestAdjugatableRow = row + 1;
                        }

                        added = false;
                        node = cols.First;
                        for (; node != null; node = node.Next)
                        {
                            if (node.Value > col)
                            {
                                ColSum += col;
                                colNode = cols.AddBefore(node, col);
                                added = true;

                                if (SmallestAdjugatableCol < col)
                                    break;
                                col++;

                                while (node != null && node.Value == col)
                                {
                                    node = node.Next;
                                    col++;
                                }
                                SmallestAdjugatableCol = col;
                                break;

                            }
                            else if (col == node.Value)
                                break;
                        }

                        if (!added)
                        {
                            ColSum += col;
                            colNode = cols.AddLast(col);

                            if (SmallestAdjugatableCol >= col)
                                SmallestAdjugatableCol = col + 1;
                        }
                        UpdateColSkip();
                    }
                }

                public static ISquareMatrix operator *(ISquareMatrix A, ISquareMatrix B) => A.Multiply(B).ToSquare;

                public static ISquareMatrix operator +(ISquareMatrix matrix, float value) => (ISquareMatrix)matrix.Add(value);
                public static ISquareMatrix operator -(ISquareMatrix matrix, float value) => (ISquareMatrix)matrix.Subtract(value);
                public static ISquareMatrix operator *(ISquareMatrix matrix, float value) => (ISquareMatrix)matrix.Multiply(value);
                public static ISquareMatrix operator /(ISquareMatrix matrix, float value) => (ISquareMatrix)matrix.Divide(value);

                public static ISquareMatrix operator +(float value, ISquareMatrix matrix) => (ISquareMatrix)matrix.Add(value);
                public static ISquareMatrix operator -(float value, ISquareMatrix matrix) => (ISquareMatrix)matrix.LeftSubtract(value);
                public static ISquareMatrix operator *(float value, ISquareMatrix matrix) => (ISquareMatrix)matrix.Multiply(value);
            }

            public class DenseMatrix : IMatrix
            {
                public float[][] content;
                public int rowCount => content.Length;
                public int colCount => content[0].Length;

                public virtual IMatrix Transpose
                {
                    get
                    {
                        IMatrix result = InstanceT();

                        for (int i = 0; i < colCount; i++)
                            for (int j = 0; j < rowCount; j++)
                                result.Set(i, j, Get(j, i));

                        return result;
                    }
                }

                public virtual ISquareMatrix ToSquare
                {
                    get
                    {
                        if (this is ISquareMatrix)
                            return (ISquareMatrix)Clone();

                        ISquareMatrix result = new DenseSquareMatrix(rowCount);
                        result.SetTo(this);
                        return result;
                    }
                }

                public DenseMatrix(int row, int col)
                {
                    content = new float[row][];

                    for (int i = 0; i < row; i++)
                        content[i] = new float[col];
                }

                public DenseMatrix(float[][] content)
                {
                    this.content = content;
                }

                public virtual IMatrix Instance() => new DenseMatrix(rowCount, colCount);

                public virtual IMatrix InstanceT() => new DenseMatrix(colCount, rowCount);

                public virtual float Get(int row, int col) => content[row][col];

                public virtual bool Set(int row, int col, float value)
                {
                    content[row][col] = value;
                    return true;
                }

                public virtual bool SetTo(IMatrix matrix)
                {
                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            Set(i, j, matrix.Get(i, j));

                    return true;
                }

                public virtual IMatrix Clone()
                {
                    IMatrix result = Instance();
                    result.SetTo(this);

                    return result;
                }

                public virtual Vector Multiply(Vector vector)
                {
                    if (colCount != vector.dim)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    float[] result = new float[rowCount];

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result[i] += Get(i, j) * vector.content[j];

                    return result;
                }

                public virtual Vector LeftMultiply(Vector vector)
                {
                    if (rowCount != vector.dim)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    float[] result = new float[colCount];

                    for (int i = 0; i < colCount; i++)
                        for (int j = 0; j < rowCount; j++)
                            result[i] += Get(i, j) * vector.content[j];

                    return result;
                }

                public virtual IMatrix Multiply(IMatrix matrix)
                {
                    if (colCount != matrix.rowCount)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    IMatrix result = new DenseMatrix(rowCount, matrix.colCount);

                    for (int row = 0; row < rowCount; row++)
                        for (int col = 0; col < matrix.colCount; col++)
                            for (int i = 0; i < colCount; i++)
                                result.Set(row, col, result.Get(row, col) + Get(row, i) * matrix.Get(i, col));

                    return result;
                }

                public virtual IMatrix Add(float value)
                {
                    IMatrix result = Instance();

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result.Set(i, j, Get(i, j) + value);

                    return result;
                }

                public virtual IMatrix Subtract(float value)
                {
                    return Add(-value);
                }

                public virtual IMatrix LeftSubtract(float value)
                {
                    IMatrix result = Instance();

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result.Set(i, j, value - Get(i, j));

                    return result;
                }

                public virtual IMatrix Multiply(float value)
                {
                    IMatrix result = Instance();

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result.Set(i, j, Get(i, j) * value);

                    return result;
                }

                public virtual IMatrix Divide(float value)
                {
                    IMatrix result = Instance();

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result.Set(i, j, Get(i, j) / value);

                    return result;
                }
            }

            public class DenseSquareMatrix : DenseMatrix, ISquareMatrix
            {
                public int dim => content.Length;

                public DenseSquareMatrix(int dim) : base(dim, dim) { }

                public DenseSquareMatrix(float[][] content) : base(content)
                {
                    if (content.Length != content[0].Length)
                        throw new Exception("Invalid square matrix content");
                }

                public override IMatrix Instance() => new DenseSquareMatrix(dim);
                public override IMatrix InstanceT() => new DenseSquareMatrix(dim);

                public virtual ISquareMatrix Invert()
                {
                    return (ISquareMatrix)Adjugate().Divide(Determinant());
                }

                public virtual float Determinant()
                {
                    if (dim == 1)
                        return content[0][0];
                    else if (dim == 2)
                        return content[0][0] * content[1][1] - content[1][0] * content[0][1];
                    else
                    {
                        float result = 0;

                        for (int i = 0; i < dim; i++)
                            result += Cofactor(i, 0) * content[i][0];

                        return result;
                    }
                }

                public virtual float Cofactor(int row, int col)
                {
                    return ISquareMatrix.Cofactor(this, row, col);
                }

                public virtual ISquareMatrix Adjugate()
                {
                    ISquareMatrix result = (ISquareMatrix)Instance();

                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            result.Set(i, j, Cofactor(j, i));

                    return result;
                }
            }

            public class DiagonalMatrix : ISquareMatrix
            {
                public float[] content;

                public int dim => content.Length;
                public int rowCount => content.Length;
                public int colCount => content.Length;

                public IMatrix Transpose => Clone();

                public ISquareMatrix ToSquare => (ISquareMatrix)Clone();

                public DiagonalMatrix(int dim)
                {
                    content = new float[dim];
                }

                public DiagonalMatrix(float[] content)
                {
                    this.content = content;
                }

                public virtual IMatrix Instance() => new DiagonalMatrix(dim);
                public virtual IMatrix InstanceT() => Instance();

                public virtual float Get(int row, int col)
                {
                    if (row >= rowCount || col >= colCount)
                        throw new Exception("Matrix entry out of bound");

                    return row == col ? content[row] : 0;
                }

                public virtual bool Set(int row, int col, float value)
                {
                    if (row >= rowCount || col >= colCount)
                        throw new Exception("Matrix entry out of bound");

                    if (row != col)
                        return false;

                    content[row] = value;
                    return true;
                }

                public virtual bool SetTo(IMatrix matrix)
                {
                    if (!(matrix is DiagonalMatrix) || matrix.rowCount != dim)
                        return false;

                    for (int i = 0; i < dim; i++)
                        Set(i, i, matrix.Get(i, i));
                    return true;
                }

                public virtual IMatrix Clone()
                {
                    IMatrix matrix = Instance();
                    matrix.SetTo(this);
                    return matrix;
                }

                public virtual ISquareMatrix Invert()
                {
                    ISquareMatrix result = (ISquareMatrix)Instance();

                    for (int i = 0; i < dim; i++)
                        result.Set(i, i, 1 / Get(i, i));

                    return result;
                }

                public virtual float Determinant()
                {
                    float result = 1;
                    for (int i = 0; i < dim; i++)
                        result *= Get(i, i);

                    return result;
                }

                public virtual float Cofactor(int row, int col)
                {
                    if (row != col)
                        return 0;

                    float result = 1;

                    for (int i = 0; i < row; i++)
                        result *= Get(i, i);
                    for (int i = row + 1; i < dim; i++)
                        result *= Get(i, i);

                    return result;
                }

                public virtual ISquareMatrix Adjugate()
                {
                    ISquareMatrix result = (ISquareMatrix)Instance();
                    for (int i = 0; i < dim; i++)
                        result.Set(i, i, Cofactor(i, i));

                    return result;
                }

                public virtual Vector Multiply(Vector vector)
                {
                    if (colCount != vector.dim)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    float[] result = new float[rowCount];

                    for (int i = 0; i < dim; i++)
                        result[i] += Get(i, i) * vector.content[i];

                    return result;
                }

                public virtual Vector LeftMultiply(Vector vector)
                {
                    if (rowCount != vector.dim)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    float[] result = new float[colCount];

                    for (int i = 0; i < dim; i++)
                        result[i] += Get(i, i) * vector.content[i];

                    return result;
                }

                public virtual IMatrix Multiply(IMatrix matrix)
                {
                    if (colCount != matrix.rowCount)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    IMatrix result = new DenseMatrix(rowCount, matrix.colCount);

                    for (int row = 0; row < rowCount; row++)
                        for (int col = 0; col < matrix.colCount; col++)
                            result.Set(row, col, Get(row, col) * matrix.Get(col, col));

                    return result;
                }

                public virtual IMatrix Add(float value)
                {
                    IMatrix result = new DenseSquareMatrix(dim);

                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            result.Set(i, j, Get(i, j) + value);

                    return result;
                }

                public virtual IMatrix Subtract(float value)
                {
                    return Add(-value);
                }

                public virtual IMatrix LeftSubtract(float value)
                {
                    IMatrix result = new DenseSquareMatrix(dim);

                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                            result.Set(i, j, value - Get(i, j));

                    return result;
                }

                public virtual IMatrix Multiply(float value)
                {
                    IMatrix result = Instance();

                    for (int i = 0; i < dim; i++)
                        result.Set(i, i, Get(i, i) * value);

                    return result;
                }

                public virtual IMatrix Divide(float value)
                {
                    IMatrix result = Instance();

                    for (int i = 0; i < dim; i++)
                        result.Set(i, i, Get(i, i) / value);

                    return result;
                }
            }

            public class TriangularMatrix : ISquareMatrix
            {
                public float[] content;
                public bool isUpper;

                public int dim { get; protected set; }
                public int rowCount => dim;
                public int colCount => dim;

                public virtual IMatrix Transpose
                {
                    get
                    {
                        TriangularMatrix result = (TriangularMatrix)Clone();
                        result.isUpper = !isUpper;
                        return result;
                    }
                }

                public ISquareMatrix ToSquare => (ISquareMatrix)Clone();

                public TriangularMatrix(int dim, bool isUpper)
                {
                    content = new float[(dim * (1 + dim)) >> 1];
                    this.isUpper = isUpper;
                    this.dim = dim;
                }

                public TriangularMatrix(float[] content, bool isUpper)
                {
                    this.content = content;
                    this.isUpper = isUpper;
                    this.dim = (int)(MathF.Sqrt(1 + 8 * content.Length) - 1) >> 1;
                }

                public virtual IMatrix Instance() => new TriangularMatrix(dim, isUpper);
                public virtual IMatrix InstanceT() => new TriangularMatrix(dim, !isUpper);

                public virtual float Get(int row, int col)
                {
                    if (row >= rowCount || col >= colCount)
                        throw new Exception("Matrix entry out of bound");

                    if ((row > col) == isUpper && row != col)
                        return 0;

                    if (isUpper)
                        return content[((col * (1 + col)) >> 1) + row];
                    else
                        return content[((row * (1 + row)) >> 1) + col];
                }

                public virtual bool Set(int row, int col, float value)
                {
                    if (row >= rowCount || col >= colCount)
                        throw new Exception("Matrix entry out of bound");

                    if ((row > col) == isUpper && row != col)
                        return false;

                    if (isUpper)
                        content[((col * (1 + col)) >> 1) + row] = value;
                    else
                        content[((row * (1 + row)) >> 1) + col] = value;

                    return true;
                }

                public virtual Vector Substitute(Vector rhs)
                {
                    if (rhs.dim != dim)
                        throw new Exception("Invalid inputs for triangular substitution");

                    Vector result = new Vector(dim);

                    if (isUpper)
                    {
                        for (int i = dim - 1; i >= 0; i--)
                        {
                            result.content[i] = rhs.content[i];
                            for (int j = i + 1; j < dim; j++)
                                result.content[i] -= result.content[j] * Get(i, j);
                            result.content[i] /= Get(i, i);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < dim; i++)
                        {
                            result.content[i] = rhs.content[i];
                            for (int j = 0; j < i; j++)
                                result.content[i] -= result.content[j] * Get(i, j);
                            result.content[i] /= Get(i, i);
                        }
                    }

                    return result;
                }

                public virtual bool SetTo(IMatrix matrix)
                {
                    if (!(matrix is DiagonalMatrix || matrix is TriangularMatrix) || matrix.rowCount != dim)
                        return false;

                    if (isUpper)
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = i; j < dim; j++)
                                Set(i, j, matrix.Get(i, j));
                    }
                    else
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = 0; j <= i; j++)
                                Set(i, j, matrix.Get(i, j));
                    }

                    return true;
                }

                public virtual IMatrix Clone()
                {
                    IMatrix matrix = Instance();
                    matrix.SetTo(this);
                    return matrix;
                }

                public virtual ISquareMatrix Invert()
                {
                    ISquareMatrix result = (ISquareMatrix)Instance();
                    float det = Determinant();

                    if (isUpper)
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = i; j < dim; j++)
                                result.Set(i, j, IMatrix.NegOneRaiseTo(i + j) * Cofactor(j, i) / det);
                    }
                    else
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = 0; j <= i; j++)
                                result.Set(i, j, IMatrix.NegOneRaiseTo(i + j) * Cofactor(j, i) / det);
                    }

                    return result;
                }

                public virtual float Determinant()
                {
                    float result = 1;
                    for (int i = 0; i < dim; i++)
                        result *= Get(i, i);

                    return result;
                }

                public virtual float Cofactor(int row, int col)
                {
                    if (row > col != isUpper)
                        return 0;

                    if (row == col)
                        return Determinant() / Get(row, col);

                    return ISquareMatrix.Cofactor(this, row, col);
                }

                public virtual ISquareMatrix Adjugate()
                {
                    ISquareMatrix result = (ISquareMatrix)Instance();

                    if (isUpper)
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = i; j < dim; j++)
                                result.Set(i, j, IMatrix.NegOneRaiseTo(i + j) * Cofactor(j, i));
                    }
                    else
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = 0; j <= i; j++)
                                result.Set(i, j, IMatrix.NegOneRaiseTo(i + j) * Cofactor(j, i));
                    }

                    return result;
                }

                public virtual Vector Multiply(Vector vector)
                {
                    if (dim != vector.dim)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    float[] result = new float[dim];

                    if (isUpper)
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = i; j < dim; j++)
                                result[i] += Get(i, j) * vector.content[j];
                    }
                    else
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = 0; j <= i; j++)
                                result[i] += Get(i, j) * vector.content[j];
                    }

                    return result;
                }

                public virtual Vector LeftMultiply(Vector vector)
                {
                    if (dim != vector.dim)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    float[] result = new float[dim];

                    if (isUpper)
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = i; j < dim; j++)
                                result[j] += Get(i, j) * vector.content[i];
                    }
                    else
                    {
                        for (int i = 0; i < dim; i++)
                            for (int j = 0; j <= i; j++)
                                result[j] += Get(i, j) * vector.content[i];
                    }

                    return result;
                }

                public virtual IMatrix Multiply(IMatrix matrix)
                {
                    if (colCount != matrix.rowCount)
                        throw new Exception("Invalid input matrices for matrix multiplication");

                    IMatrix result = new DenseMatrix(rowCount, matrix.colCount);

                    if (isUpper)
                    {
                        for (int row = 0; row < rowCount; row++)
                            for (int col = 0; col < matrix.colCount; col++)
                                for (int i = row; i < colCount; i++)
                                    result.Set(row, col, result.Get(row, col) + Get(row, i) * matrix.Get(i, col));
                    }
                    else
                    {
                        for (int row = 0; row < rowCount; row++)
                            for (int col = 0; col < matrix.colCount; col++)
                                for (int i = 0; i <= row; i++)
                                    result.Set(row, col, result.Get(row, col) + Get(row, i) * matrix.Get(i, col));
                    }

                    return result;
                }

                public virtual IMatrix Add(float value)
                {
                    IMatrix result = new DenseSquareMatrix(dim);

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result.Set(i, j, Get(i, j) + value);

                    return result;
                }

                public virtual IMatrix Subtract(float value)
                {
                    return Add(-value);
                }

                public virtual IMatrix LeftSubtract(float value)
                {
                    IMatrix result = new DenseSquareMatrix(dim);

                    for (int i = 0; i < rowCount; i++)
                        for (int j = 0; j < colCount; j++)
                            result.Set(i, j, value - Get(i, j));

                    return result;
                }

                public virtual IMatrix Multiply(float value)
                {
                    TriangularMatrix result = (TriangularMatrix)Instance();

                    for (int i = 0; i < content.Length; i++)
                        result.content[i] = content[i] * value;

                    return result;
                }

                public virtual IMatrix Divide(float value)
                {
                    TriangularMatrix result = (TriangularMatrix)Instance();

                    for (int i = 0; i < content.Length; i++)
                        result.content[i] = content[i] / value;

                    return result;
                }
            }

            public class Vector
            {
                public float[] content;

                public int dim => content.Length;

                public IMatrix ToMatrix
                {
                    get
                    {
                        IMatrix result = new DenseMatrix(1, dim);
                        for (int i = 0; i < dim; i++)
                            result.Set(0, i, content[i]);

                        return result;
                    }
                }

                public Vector(float[] content)
                {
                    this.content = content;
                }

                public Vector(int size)
                {
                    this.content = new float[size];
                }

                public void SetTo(Vector vector)
                {
                    for (int i = 0; i < dim; i++)
                        content[i] = vector.content[i];
                }

                public Vector Clone()
                {
                    Vector clone = new Vector(dim);

                    for (int i = 0; i < dim; i++)
                        clone.content[i] = content[i];

                    return clone;
                }

                public static Vector Clone(float[] content)
                {
                    Vector clone = new Vector(content.Length);

                    for (int i = 0; i < clone.dim; i++)
                        clone.content[i] = content[i];

                    return clone;
                }

                public static float[] Add(float[] a, float[] b)
                {
                    if (a.LongLength != b.LongLength)
                        throw new Exception("Invalid input vectors for dot product");

                    float[] result = new float[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] + b[i];

                    return result;
                }

                public static float[] Add(float[] a, float b)
                {
                    float[] result = new float[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] + b;

                    return result;
                }

                public static float[] Add(float b, float[] a)
                {
                    float[] result = new float[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] + b;

                    return result;
                }

                public static float[] Subtract(float[] a, float[] b)
                {
                    if (a.LongLength != b.LongLength)
                        throw new Exception("Invalid input vectors for dot product");

                    float[] result = new float[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] - b[i];

                    return result;
                }

                public static float[] Subtract(float[] a, float b)
                {
                    float[] result = new float[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] - b;

                    return result;
                }

                public static float[] Subtract(float b, float[] a)
                {
                    float[] result = new float[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = b - a[i];

                    return result;
                }

                public static float[] Multiply(float[] a, float b)
                {
                    float[] result = new float[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] * b;

                    return result;
                }

                public static float[] Divide(float[] a, float b)
                {
                    float[] result = new float[a.LongLength];

                    for (int i = 0; i < a.LongLength; i++)
                        result[i] = a[i] / b;

                    return result;
                }

                public static float Dot(float[] a, float[] b)
                {
                    if (a.LongLength != b.LongLength)
                        throw new Exception("Invalid input vectors for dot product");

                    float result = 0;

                    for (int i = 0; i < a.LongLength; i++)
                        result += a[i] * b[i];

                    return result;
                }

                public static float Dot(Vector a, Vector b) => Dot(a.content, b.content);

                public static implicit operator Vector(float[] content) => new Vector(content);

                public static Vector operator +(Vector a, Vector b) => Add(a.content, b.content);
                public static Vector operator +(Vector a, float b) => Add(a.content, b);
                public static Vector operator +(float b, Vector a) => Add(a.content, b);

                public static Vector operator -(Vector a, Vector b) => Subtract(a.content, b.content);
                public static Vector operator -(Vector a, float b) => Subtract(a.content, b);
                public static Vector operator -(float b, Vector a) => Subtract(b, a.content);

                public static Vector operator *(Vector a, float b) => Multiply(a.content, b);
                public static Vector operator *(float b, Vector a) => Multiply(a.content, b);
                public static Vector operator /(Vector a, float b) => Divide(a.content, b);
            }
        }
    }
}