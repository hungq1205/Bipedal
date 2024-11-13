using Lib.AI.RL;
using System.Collections.Generic;
using System.IO;
using System;
using System.Linq;
using System.Text;
using System.Reflection;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using static Lib.Utility.Common;

namespace Lib
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

            void IPolicy.CloneTo(IPolicy policy)
            {
                var nn = (INeuralNetwork)policy;

                for (int i = 0; i < Layers.Length; i++)
                    Layers[i].CloneTo(nn.Layers[i]);

                for (int i = 0; i < Weights.Length; i++)
                    Weights[i].CloneTo(nn.Weights[i]);
            }
        }

        public class DenseNeuralNetwork : INeuralNetwork
        {
            public int InDim { get; private set; }
            public int OutDim { get; private set; }
            public Optimizer Optimizer { get; private set; }
            public Layer[] Layers { get; private set; }
            public WeightMatrix[] Weights { get; private set; }

            public ForwardResult Log { get; set; }

            public DenseNeuralNetwork() { }

            private DenseNeuralNetwork(DenseNeuralNetwork source)
            {
                InDim = source.InDim;
                OutDim = source.OutDim;

                Layers = new Layer[source.Layers.Length];
                for (int i = 0; i < source.Layers.Length; i++)
                {
                    Layers[i] = source.Layers[i].Clone();
                    Layers[i].Build(this);
                    source.Layers[i].CloneTo(Layers[i]);
                }

                Weights = new WeightMatrix[source.Weights.Length];
                for (int i = 0; i < source.Weights.Length; i++)
                {
                    Weights[i] = source.Weights[i].Clone();
                    Weights[i].Build(this);
                    source.Weights[i].CloneTo(Weights[i]);
                }
            }

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
                    Weights[i].AssignForEach((inIndex, outIndex, weight) => func(weight, Weights[i].InDim, Weights[i].OutDim));
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

            public float[] Infer(float[] X)
            {
                return InferLayers(X, Layers.Length - 1, 0);
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

            float[] InferLayers(float[] X, int toLayer, int fromLayer)
            {
                if (fromLayer < toLayer)
                    X = Weights[toLayer - 1].Forward(InferLayers(X, toLayer - 1, fromLayer));
                return Layers[toLayer].Forward(X);
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

            public IPolicy Clone()
            {
                return new DenseNeuralNetwork(this);
            }

            public void Save(string filepath)
            {
                using var fileStream = new StreamWriter(filepath);
                using var writer = new JsonTextWriter(fileStream);

                writer.WriteStartObject();
                writer.WritePropertyName("Layers");
                writer.WriteStartArray();

                for (int i = 0; i < Layers.Length; i++)
                    Layer.ToJson(writer, Layers[i]);

                writer.WriteEndArray();
                writer.WritePropertyName("Weights");
                writer.WriteStartArray();

                for (int i = 0; i < Weights.Length; i++)
                    WeightMatrix.ToJson(writer, Weights[i]);

                writer.WriteEndArray();
                writer.WritePropertyName("Optimizer");
                Optimizer.ToJson(writer, Optimizer);

                writer.WriteEndObject();
            }

            public void Load(string filepath)
            {
                var jobject = JObject.Parse(File.ReadAllText(filepath));

                var layers = new List<Layer>();
                foreach (var layer in jobject["Layers"].Children())
                    layers.Add(Layer.FromJson(layer.ToString()));

                var weights = new List<WeightMatrix>();
                foreach (var weight in jobject["Weights"].Children())
                    weights.Add(WeightMatrix.FromJson(weight.ToString()));

                Optimizer = Optimizer.FromJson(jobject["Optimizer"].ToString());

                Layers = layers.ToArray();
                Weights = weights.ToArray();

                InDim = Layers[0].dim;
                OutDim = Layers[Layers.LongLength - 1].dim;

                foreach (var layer in Layers)
                    layer.Build(this);

                foreach (var weight in Weights)
                    weight.Build(this);

                Optimizer.Init(this);
            }
        }

        public class DenseNeuralNetworkBuilder : INeuralNetworkBuilder, IDisposable
        {
            public List<Layer> layers;

            public DenseNeuralNetworkBuilder(int inputDim)
            {
                layers = new List<Layer>();

                layers.Add(new Layer(inputDim, true));
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

        public abstract class Optimizer : IJsonConvertable
        {
            readonly static Dictionary<string, Type> types = GetChildrenTypeOf<Optimizer>();

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

            public abstract IEnumerable<(string, object)> GetJsonProperties();

            public static void ToJson(JsonWriter writer, Optimizer opt)
                => StoreJson(writer, opt);

            public static Optimizer FromJson(string json)
                => (Optimizer)LoadJson(json, types);
        }

        public class SGD : Optimizer, IBatchNormOptimizable
        {
            public float learningRate;
            public Dictionary<int, int> bnIndexLookup { get; private set; }

            [JsonConstructor]
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

            public override IEnumerable<(string, object)> GetJsonProperties()
            {
                yield return ("learningRate", learningRate);
                yield return ("weightDecay", weightDecay);
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

            [JsonConstructor]
            private Momentum(float[][][] weightMomentum, float[][] biasMomentum, float beta = 0.9f, float learningRate = 0.01f, float weightDecay = 0) : base(weightDecay)
            {
                this.learningRate = learningRate;
                this.beta = beta;
                this.weightMomentum = weightMomentum;
                this.biasMomentum = biasMomentum;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;

                if (weightMomentum == null)
                {
                    weightMomentum = new float[network.Weights.Length][][];
                    for (int i = 0; i < network.Weights.Length; i++)
                    {
                        weightMomentum[i] = new float[network.Weights[i].OutDim][];
                        for (int j = 0; j < network.Weights[i].OutDim; j++)
                        {
                            weightMomentum[i][j] = new float[network.Weights[i].InDim];
                            for (int k = 0; k < network.Weights[i].InDim; k++)
                                weightMomentum[i][j][k] = 0.000001f; // epsilon = 10^-6
                        }
                    }
                }

                bnIndexLookup ??= new Dictionary<int, int>();

                if (biasMomentum == null)
                {
                    biasMomentum = new float[network.Layers.Length][];
                    for (int i = 0; i < network.Layers.Length; i++)
                    {
                        biasMomentum[i] = new float[network.Layers[i].dim];
                        for (int j = 0; j < network.Layers[i].dim; j++)
                            biasMomentum[i][j] = 0.000001f; // epsilon = 10^-6

                        if (network.Layers[i] is BatchNormLayer)
                            bnIndexLookup.Add(i, bnIndexLookup.Count);
                    }
                }    

                gammaMomentum ??= new float[bnIndexLookup.Count];
                betaMomentum ??= new float[bnIndexLookup.Count];
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

            public override IEnumerable<(string, object)> GetJsonProperties()
            {
                yield return ("weightMomentum", weightMomentum);
                yield return ("biasMomentum", biasMomentum);
                yield return ("beta", beta);
                yield return ("learningRate", learningRate);
                yield return ("weightDecay", weightDecay);
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
                    accumWeightGrad[i] = new float[network.Weights[i].OutDim][];
                    for (int j = 0; j < network.Weights[i].OutDim; j++)
                    {
                        accumWeightGrad[i][j] = new float[network.Weights[i].InDim];
                        for (int k = 0; k < network.Weights[i].InDim; k++)
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

            public override IEnumerable<(string, object)> GetJsonProperties()
            {
                yield return ("learningRate", learningRate);
                yield return ("weightDecay", weightDecay);
            }
        }

        public class Adam : Optimizer, IBatchNormOptimizable
        {
            public float[][][] accumWeightGrad, weightMomentum;
            public float[][] accumBiasGrad, biasMomentum;
            public float learningRate, beta1, beta2, epsilon;

            public Dictionary<int, int> bnIndexLookup { get; private set; }
            public float[] accumGammaGrad, gammaMomentum, accumBetaGrad, betaMomentum;

            public Adam(float beta1 = 0.9f, float beta2 = 0.99f, float learningRate = 0.01f, float weightDecay = 0, float epsilon = 1e-2f) : base(weightDecay)
            {
                this.learningRate = learningRate;
                this.beta1 = beta1;
                this.beta2 = beta2;
                this.epsilon = epsilon;
            }

            public override void Init(DenseNeuralNetwork network)
            {
                this.network = network;

                weightMomentum = new float[network.Weights.Length][][];
                accumWeightGrad = new float[network.Weights.Length][][];
                for (int i = 0; i < network.Weights.Length; i++)
                {
                    weightMomentum[i] = new float[network.Weights[i].OutDim][];
                    accumWeightGrad[i] = new float[network.Weights[i].OutDim][];
                    for (int j = 0; j < network.Weights[i].OutDim; j++)
                    {
                        weightMomentum[i][j] = new float[network.Weights[i].InDim];
                        accumWeightGrad[i][j] = new float[network.Weights[i].InDim];
                        for (int k = 0; k < network.Weights[i].InDim; k++)
                        {
                            weightMomentum[i][j][k] = epsilon; 
                            accumWeightGrad[i][j][k] = epsilon; 
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
                        biasMomentum[i][j] = epsilon; 
                        accumBiasGrad[i][j] = epsilon; 
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

                return network.Weights[weightsIndex].GetWeight(inIndex, outIndex) - (learningRate * weightMomentum[weightsIndex][outIndex][inIndex]) / OffsetZero(MathF.Sqrt(accumWeightGrad[weightsIndex][outIndex][inIndex]));
            }

            public override float BiasUpdate(int layerIndex, int perceptron, float gradient)
            {
                biasMomentum[layerIndex][perceptron] = beta1 * biasMomentum[layerIndex][perceptron] + (1 - beta1) * gradient;
                accumBiasGrad[layerIndex][perceptron] = beta2 * accumBiasGrad[layerIndex][perceptron] + (1 - beta2) * gradient * gradient;

                return network.Layers[layerIndex].GetBias(perceptron) - (learningRate * biasMomentum[layerIndex][perceptron]) / OffsetZero(MathF.Sqrt(accumBiasGrad[layerIndex][perceptron]));
            }

            public float GammaUpdate(int layerIndex, float gradient)
            {
                gammaMomentum[bnIndexLookup[layerIndex]] = beta1 * gammaMomentum[bnIndexLookup[layerIndex]] + (1 - beta1) * gradient;
                accumGammaGrad[bnIndexLookup[layerIndex]] = beta2 * accumGammaGrad[bnIndexLookup[layerIndex]] + (1 - beta2) * gradient * gradient;

                return ((BatchNormLayer)network.Layers[layerIndex]).gamma - (learningRate * gammaMomentum[bnIndexLookup[layerIndex]]) / OffsetZero(MathF.Sqrt(accumGammaGrad[bnIndexLookup[layerIndex]]));
            }

            public float BetaUpdate(int layerIndex, float gradient)
            {
                betaMomentum[bnIndexLookup[layerIndex]] = beta1 * betaMomentum[bnIndexLookup[layerIndex]] + (1 - beta1) * gradient;
                accumBetaGrad[bnIndexLookup[layerIndex]] = beta2 * accumBetaGrad[bnIndexLookup[layerIndex]] + (1 - beta2) * gradient * gradient;

                return ((BatchNormLayer)network.Layers[layerIndex]).beta - (learningRate * betaMomentum[bnIndexLookup[layerIndex]]) / OffsetZero(MathF.Sqrt(accumBetaGrad[bnIndexLookup[layerIndex]]));
            }

            float OffsetZero(float value)
            {
                float rs = value;
                if (MathF.Abs(value) < epsilon)
                    rs = MathF.Sign(value) * epsilon;
                return rs;
            }

            public override IEnumerable<(string, object)> GetJsonProperties()
            {
                yield return ("learningRate", learningRate);
                yield return ("weightDecay", weightDecay);
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
                    accumWeightGrad[i] = new float[network.Weights[i].OutDim][];
                    for (int j = 0; j < network.Weights[i].OutDim; j++)
                    {
                        accumWeightGrad[i][j] = new float[network.Weights[i].InDim];
                        for (int k = 0; k < network.Weights[i].InDim; k++)
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

            public override IEnumerable<(string, object)> GetJsonProperties()
            {
                yield return ("weightDecay", weightDecay);
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
                    accumWeightGrad[i] = new float[network.Weights[i].OutDim][];
                    accumRescaledWeightGrad[i] = new float[network.Weights[i].OutDim][];
                    for (int j = 0; j < network.Weights[i].OutDim; j++)
                    {
                        accumWeightGrad[i][j] = new float[network.Weights[i].InDim];
                        accumRescaledWeightGrad[i][j] = new float[network.Weights[i].InDim];
                        for (int k = 0; k < network.Weights[i].InDim; k++)
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

            public override IEnumerable<(string, object)> GetJsonProperties()
            {
                yield return ("weightDecay", weightDecay);
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

        public class BatchNormLayer : ActivationForwardLayer
        {
            public float gamma = 1, beta = 0;

            public BatchNormLayer(ForwardLayer.ForwardPort port) : base(ActivationFunc.Custom, port, false) { }

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

        public class NormalizationLayer : ActivationForwardLayer
        {
            public float gamma, beta;

            public NormalizationLayer(float min, float max, ForwardLayer.ForwardPort port) : base(ActivationFunc.Custom, port, false)
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

        public class Dropout : ForwardLayer
        {
            public float rate;

            bool[] drops;

            [JsonConstructor]
            public Dropout(float rate) : base(1f, 0f, ForwardPort.In, false)
            {
                this.rate = rate;
            }

            public override void Build(INeuralNetwork network)
            {
                base.Build(network);
                drops = new bool[dim];
            }

            public override float[] Forward(float[] X)
            {
                float[] result = new float[dim];
                var rand = new Random();

                for (int i = 0; i < dim; i++)
                {
                    drops[i] = (float)rand.NextDouble() < rate;
                    result[i] = drops[i] ? 0f : X[i];
                }

                return result;
            }

            public override float[] Infer(float[] X)
            {
                float[] Y = new float[dim];

                for (int i = 0; i < dim; i++)
                    Y[i] = X[i];

                return Y;
            }

            public override float[] FunctionDifferential(float[] X, float[] loss)
            {
                float[] result = new float[X.Length];
                for (int i = 0; i < X.Length; i++)
                    result[i] = drops[i] ? 0f : loss[i];

                return result;
            }

            public override IEnumerable<(string,object)> GetJsonProperties()
            {
                yield return ("rate", rate);
            }
        }

        public class ActivationForwardLayer : ActivationLayer
        {
            public readonly ForwardLayer.ForwardPort port;

            public ActivationForwardLayer(ActivationFunc func, ForwardLayer.ForwardPort port, bool useBias = true) : base(-1, func, useBias)
            {
                this.port = port;
            }

            [JsonConstructor]
            protected ActivationForwardLayer(float[] biases, bool useBias, ActivationFunc func, ForwardLayer.ForwardPort port) : base(biases, useBias, func)
            {
                this.port = port;
            }

            public override void Build(INeuralNetwork network)
            {
                base.Build(network);

                switch (port)
                {
                    case ForwardLayer.ForwardPort.In:
                        dim = network.Layers[layerIndex - 1].dim;
                        break;
                    case ForwardLayer.ForwardPort.Out:
                        dim = network.Layers[layerIndex + 1].dim;
                        break;
                    case ForwardLayer.ForwardPort.Both:
                        if (network.Layers[layerIndex - 1].dim != network.Layers[layerIndex + 1].dim)
                            throw new Exception("Nah forward layer dim");
                        dim = network.Layers[layerIndex + 1].dim;
                        break;
                }

                biases ??= new float[dim];
            }

            public override IEnumerable<(string,object)> GetJsonProperties()
            {
                yield return ("biases", biases);
                yield return ("useBias", useBias);
                yield return ("func", func);
                yield return ("port", port);
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

            [JsonConstructor]
            protected ActivationLayer(float[] biases, bool useBias, ActivationFunc func) : base(biases, useBias)
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

            public override Layer Clone()
            {
                var clone = new ActivationLayer(dim, func, useBias);
                CloneTo(clone);
                return clone;
            }

            public override IEnumerable<(string,object)> GetJsonProperties()
            {
                yield return ("biases", biases);
                yield return ("useBias", useBias);
                yield return ("func", func);
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
                        float min = X[0];

                        for (int i = 1; i < X.Length; i++)
                            if (min > X[i])
                                min = X[i];

                        for (int i = 0; i < X.Length; i++)
                        {
                            result[i] = MathF.Exp(X[i] - min);
                            temp += result[i];
                        }
                        temp = 1f / temp;
                        for (int i = 0; i < X.Length; i++)
                        {
                            result[i] *= temp;
                            if (result[i] is float.NaN)
                                result[i] = 0;
                        }
                        UnityEngine.Debug.Log(ArrayToString(X) + "\n\t  " + ArrayToString(result));
                        break;
                    case ActivationFunc.Linear:
                    default:
                        return X;
                }

                return result;
            }
        }

        public class ForwardLayer : Layer
        {
            public enum ForwardPort
            {
                In,
                Out,
                Both
            }

            public readonly ForwardPort port;

            float w, b;

            public ForwardLayer(float w, float b, ForwardPort port = ForwardPort.In, bool useBias = false) : base(-1, useBias)
            {
                this.port = port;
                this.w = w;
                this.b = b;
            }

            [JsonConstructor]
            protected ForwardLayer(float[] biases, bool useBias, float w, float b, ForwardPort port) : base(biases, useBias)
            {
                this.port = port;
                this.w = w;
                this.b = b;
            }

            public override IEnumerable<(string,object)> GetJsonProperties()
            {
                yield return ("biases", biases);
                yield return ("useBias", useBias);
                yield return ("w", w);
                yield return ("b", b);
                yield return ("port", port);
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

                if (biases == null)
                {
                    biases = new float[dim];
                    for (int i = 0; i < dim; i++)
                        biases[i] = b;
                }
            }

            public override WeightMatrix GenerateWeightMatrix()
            {
                return new ForwardWeightMatrix(w, useBias);
            }
        }

        public class Layer : IJsonConvertable
        {
            static readonly Dictionary<string, Type> layerTypes = GetChildrenTypeOf<Layer>();

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

            [JsonConstructor]
            protected Layer(float[] biases, bool useBias)
            {
                this.dim = biases.Length;
                this.useBias = useBias;
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

            public virtual float GetBias(int index) => biases[index];

            public virtual void SetBias(int index, float value) => biases[index] = useBias ? value : biases[index];

            /// <returns>Returns descended errors</returns>
            public virtual void GradientDescent(ref float[][] errors, ForwardResult log, Optimizer optimizer)
            {
                for (int sample = 0; sample < errors.Length; sample++)
                {
                    errors[sample] = FunctionDifferential(log.layerInputs[layerIndex][sample], errors[sample]);

                    if (useBias)
                        for (int i = 0; i < dim; i++)
                            SetBias(i, optimizer.BiasUpdate(layerIndex, i, errors[sample][i]));
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

            public virtual float[] Infer(float[] X)
            {
                return Forward(X);
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

            public virtual void CloneTo(Layer layer)
            {
                for (int i = 0; i < dim; i++)
                    layer.SetBias(i, biases[i]);
            }
            public virtual Layer Clone()
            {
                return useBias ? new Layer(biases) : new Layer(dim, false);
            }

            public virtual IEnumerable<(string, object)> GetJsonProperties()
            {
                yield return ("biases", biases);
                yield return ("useBias", useBias);
            }

            public static void ToJson(JsonWriter writer, Layer layer)
                => StoreJson(writer, layer);

            public static Layer FromJson(string json)
                => (Layer)LoadJson(json, layerTypes);

            public static implicit operator Layer(int dim) => new Layer(dim);
        }

        #endregion

        #region Weight matrix

        public abstract class WeightMatrix : IJsonConvertable
        {
            readonly static Dictionary<string, Type> weightTypes = GetChildrenTypeOf<WeightMatrix>();

            public int InDim { get; protected set; }
            public int OutDim { get; protected set; }
            public INeuralNetwork Network { get; protected set; }

            protected int weightsIndex;

            public virtual void Build(INeuralNetwork network)
            {
                this.Network = network;
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

            public abstract void CloneTo(WeightMatrix weights);

            public abstract WeightMatrix Clone();

            public abstract IEnumerable<(string,object)> GetJsonProperties();

            public static void ToJson(JsonWriter writer, WeightMatrix weights)
                => StoreJson(writer, weights);

            public static WeightMatrix FromJson(string json)
                => (WeightMatrix)LoadJson(json, weightTypes);
        }

        public class ForwardWeightMatrix : WeightMatrix
        {
            public readonly bool useWeights;

            public float[] matrix;

            public int dim => InDim;

            float w = float.NaN;

            public ForwardWeightMatrix(bool useWeights = true)
            {
                this.useWeights = useWeights;
            }

            public ForwardWeightMatrix(float w, bool useWeights = false)
            {
                this.useWeights = useWeights;
                this.w = w;
            }

            [JsonConstructor]
            private ForwardWeightMatrix(float[] matrix, bool useWeights, float w)
            {
                this.matrix = matrix;
                this.w = w;
                this.useWeights = useWeights;
            }

            public override void Build(INeuralNetwork network)
            {
                base.Build(network);

                if (network.Layers[weightsIndex] is ForwardLayer && ((ForwardLayer)network.Layers[weightsIndex]).port != ForwardLayer.ForwardPort.In)
                    InDim = OutDim = network.Layers[weightsIndex].dim;
                else if (network.Layers[weightsIndex + 1] is ForwardLayer && ((ForwardLayer)network.Layers[weightsIndex + 1]).port != ForwardLayer.ForwardPort.Out)
                    InDim = OutDim = network.Layers[weightsIndex + 1].dim;
                else
                    throw new Exception("Nah forward weight dim");

                if (matrix == null)
                {
                    matrix = new float[dim];
                    if (!(w is float.NaN))
                        for (int i = 0; i < dim; i++)
                            matrix[i] = w;
                }
            }

            public override void AssignForEach(Func<int, int, float, float> value)
            {
                if (useWeights)
                    for (int i = 0; i < dim; i++)
                        matrix[i] = value(i, i, matrix[i]);
            }

            public override void GradientDescent(ref float[][] errors, ForwardResult log, Optimizer optimizer)
            {

                float[] weightErrorSum = new float[matrix.Length];
                for (int sample = 0; sample < errors.Length; sample++)
                {
                    float[] layerForward = Network.Layers[weightsIndex].Forward(log.layerInputs[weightsIndex][sample]);
                    float[] layerDif = Network.Layers[weightsIndex].FunctionDifferential(log.layerInputs[weightsIndex][sample], errors[sample]);

                    for (int i = 0; i < matrix.Length; i++)
                    {
                        weightErrorSum[i] += errors[sample][i] * layerForward[i];
                        errors[sample][i] = matrix[i] * layerDif[i];
                    }
                }

                if (!useWeights) return;

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
                        result[i][j] = inputs[i][j] * matrix[j];

                return result;
            }

            public override float ForwardComp(float[] inputs, int outputIndex)
            {
                return inputs[outputIndex] * matrix[outputIndex];
            }

            public override float GetWeight(int inIndex, int outIndex)
            {
                if (inIndex == outIndex && inIndex < dim)
                    return matrix[inIndex];

                throw new Exception("No weight here bro");
            }

            public override bool TryGetWeight(int inIndex, int outIndex, out float weight)
            {
                weight = matrix[inIndex];
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

            public override void CloneTo(WeightMatrix weights)
            {
                for (int i = 0; i < InDim; i++)
                    weights.TrySetWeight(i, i, matrix[i]);
            }

            public override WeightMatrix Clone()
            {
                var clone = new ForwardWeightMatrix(useWeights);
                clone.matrix = new float[InDim];
                CloneTo(clone);

                return clone;
            }

            public override IEnumerable<(string,object)> GetJsonProperties()
            {
                yield return ("matrix", matrix);
                yield return ("useWeights", useWeights);
                yield return ("w", w);
            }
        }

        public class DenseWeightMatrix : WeightMatrix
        {
            public float[,] matrix;

            public DenseWeightMatrix() { }

            [JsonConstructor]
            private DenseWeightMatrix(float[,] matrix)
            {
                this.matrix = matrix;
            }

            public override void Build(INeuralNetwork network)
            {
                base.Build(network);

                InDim = network.Layers[weightsIndex].dim;
                OutDim = network.Layers[weightsIndex + 1].dim;

                matrix ??= new float[OutDim, InDim];
            }

            public override void GradientDescent(ref float[][] errors, ForwardResult log, Optimizer optimizer)
            {
                Layer prevLayer = Network.Layers[weightsIndex];

                float[][] weightErrors = new float[errors.Length][];
                for (int i = 0; i < errors.Length; i++)
                    weightErrors[i] = new float[InDim];

                float[][] weightErrorSum = new float[OutDim][];
                for (int i = 0; i < OutDim; i++)
                    weightErrorSum[i] = new float[InDim];

                for (int sample = 0; sample < errors.Length; sample++)
                {
                    float[] layerForward = prevLayer.Forward(log.layerInputs[weightsIndex][sample]);

                    for (int i = 0; i < OutDim; i++)
                        for (int j = 0; j < InDim; j++)
                        {
                            weightErrorSum[i][j] += errors[sample][i] * layerForward[j];
                            weightErrors[sample][j] += errors[sample][i] * matrix[i, j];
                        }

                    weightErrors[sample] = prevLayer.FunctionDifferential(log.layerInputs[weightsIndex][sample], weightErrors[sample]);
                }

                for (int i = 0; i < OutDim; i++)
                    for (int j = 0; j < InDim; j++)
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
                for (int i = 0; i < OutDim; i++)
                    for (int j = 0; j < InDim; j++)
                        matrix[i, j] = value(j, i, matrix[i, j]);
            }

            public override float[] Forward(float[] inputs)
            {
                float[] result = new float[OutDim];

                for (int i = 0; i < OutDim; i++)
                    for (int j = 0; j < InDim; j++)
                        result[i] += inputs[j] * matrix[i, j];

                return result;
            }

            public override float[][] Forward(float[][] inputs)
            {
                float[][] result = new float[inputs.Length][];

                for (int i = 0; i < inputs.Length; i++)
                    result[i] = new float[OutDim];

                for (int i = 0; i < inputs.Length; i++)
                    for (int j = 0; j < OutDim; j++)
                        for (int k = 0; k < InDim; k++)
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

            public override void CloneTo(WeightMatrix weights)
            {
                for (int i = 0; i < OutDim; i++)
                    for (int j = 0; j < InDim; j++)
                        weights.TrySetWeight(j, i, matrix[i, j]);
            }

            public override WeightMatrix Clone()
            {
                var clone = new DenseWeightMatrix();
                clone.matrix = new float[OutDim, InDim];
                CloneTo(clone);

                return clone;
            }

            public override IEnumerable<(string, object)> GetJsonProperties()
            {
                yield return ("matrix", matrix);
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

                float Evaluate(IAgent agent, Record rec);

                /// <summary>
                /// Reset all including environment and agents
                /// </summary>
                void ResetStates();

                public class Record { }
            }

            public interface IAgent
            {
                IEnvironment Env { get; }
                IPolicy Policy { get; }
                IPolicyOptimization PolicyOpt { get; }
                ConcludeType ConcludedType { get; }

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

                float[] Infer(float[] obs);

                void Update(float[] loss);

                IPolicy Clone();

                void CloneTo(IPolicy policy);

                void Save(string filepath);

                void Load(string filepath);
            }

            public interface IPolicyOptimization
            {
                IPolicy Policy { get; }

                int GetAction(float[] obs);

                virtual float[][] ComputeLoss(float[][] obs, int[] actions, float[] rews, bool logits = false)
                {
                    float[][] loss = new float[obs.Length][];
                    for (int i = 0; i < loss.Length; i++)
                        loss[i] = ComputeLoss(obs[i], actions[i], rews[i], logits);

                    return loss;
                }

                float[] ComputeLoss(float[] obs, int action, float rew, bool logits = false);

                void Step();
            }

            public class Reinforce : IPolicyOptimization
            {
                public IPolicy Policy { get; private set; }

                public Reinforce(IPolicy policy)
                {
                    Policy = policy;
                }

                public virtual int GetAction(float[] obs) => DrawProbs(Policy.Forward(obs));

                public virtual float[] ComputeLoss(float[] obs, int action, float rew, bool logits = false)
                {
                    float[] outputs;
                    if (logits)
                        outputs = Policy.Forward(obs);
                    else
                        outputs = ActivationLayer.ForwardActivation(ActivationFunc.Softmax, Policy.Forward(obs));

                    for (int i = 0; i < outputs.Length; i++)
                    {
                        if (i == action)
                        {
                            var l = -(1 / outputs[i]) * rew;
                            //var abs = Math.Abs(l);
                            //if (abs < 0.3f)
                            //    l = MathF.Sign(l) * 0.3f;
                            outputs[i] = l;
                        }
                        else
                            outputs[i] = 0;
                    }

                    return outputs;
                }

                public void Step() { }
            }

            public class DeepQLearning : IPolicyOptimization
            {
                public IPolicy Policy { get; private set; }
                public IPolicy TargetPolicy { get; private set; }

                public float discountFactor;
                public int updateIteration;

                int curCount = 0;

                public DeepQLearning(IPolicy policy, float discountFactor, int updateIteration)
                {
                    Policy = policy;
                    TargetPolicy = policy.Clone();
                    ((INeuralNetwork)TargetPolicy).BiasAssignForEach((b, dim) => 0f);
                    ((INeuralNetwork)TargetPolicy).WeightAssignForEach((w, inDim, outDim) =>
                    {
                        float stddev = UnityEngine.Mathf.Sqrt(6f / inDim);
                        return UnityEngine.Random.Range(-stddev, stddev);
                    });
                    this.discountFactor = discountFactor;
                    this.updateIteration = updateIteration;
                }

                public virtual int GetAction(float[] obs)
                {
                    float[] outputs = Policy.Forward(obs);
                    int min = 0;
                    for (int i = 1; i < outputs.Length; i++)
                        if (outputs[min] > outputs[i])
                            min = i;
                    for (int i = 1; i < outputs.Length; i++)
                        outputs[i] -= outputs[min];
                    return DrawProbs(outputs);
                }

                public virtual float[] ComputeLoss(float[] obs, int action, float rew, bool logits = false)
                {
                    float[] outputs = TargetPolicy.Forward(obs);

                    int max = 0;
                    for (int i = 1; i < outputs.Length; i++)
                        if (outputs[max] < outputs[i])
                            max = i;

                    //UnityEngine.Debug.Log(string.Join(", ", Policy.Forward(obs)));
                    //UnityEngine.Debug.Log(string.Join(", ", outputs));
                    //UnityEngine.Debug.Log(max);
                    //UnityEngine.Debug.Log("rew: " + rew);
                    float[] loss = new float[outputs.Length];
                    float l = -(rew + discountFactor * outputs[max] - Policy.Forward(obs)[action]);
                    //UnityEngine.Debug.Log(l);
                    if (l > 10)
                        l = 10;
                    if (l < -5)
                        l = -5;
                    if (l is float.NaN)
                    {
                        UnityEngine.Debug.Log("fixed");
                        l = 0;
                    }
                    //UnityEngine.Debug.Log(l);
                    loss[max] = l;

                    //UnityEngine.Debug.Log("loss: " + string.Join(", ", loss));

                    return loss;
                }

                public void Step()
                {
                    if (++curCount > updateIteration)
                    {
                        TargetPolicy = Policy.Clone();
                        curCount = 0;
                    }
                }
            }

            public class ExplorationWrapper : IPolicyOptimization
            {
                public IPolicyOptimization content;
                public double exploreRate, exploreDecay, minRate;
                public int actionNum;

                public IPolicy Policy => content.Policy;

                public ExplorationWrapper(IPolicyOptimization content, float initialRate, float decay, int actionNum, float minRate)
                {
                    this.content = content;
                    this.actionNum = actionNum;
                    exploreRate = initialRate;
                    exploreDecay = decay;
                    this.minRate = minRate;
                }

                public int GetAction(float[] obs)
                {
                    int act = -1;
                    Random rand = new();
                    if (rand.NextDouble() > exploreRate)
                        act = content.GetAction(obs);
                    else
                        act = rand.Next(0, actionNum);

                    return act;
                }

                public float[] ComputeLoss(float[] obs, int action, float mass, bool logits = false)
                    => content.ComputeLoss(obs, action, mass, logits);

                public void Step()
                {
                    content.Step();
                    exploreRate *= exploreDecay;
                    if (exploreRate < minRate)
                        exploreRate = minRate;
                }
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
        public static class Common
        {
            public static readonly string classPropertyName = "ClassName";
            public static readonly string contentName = "Content";

            public interface IJsonConvertable
            {
                IEnumerable<(string, object)> GetJsonProperties();
            }

            public static void StoreJson(JsonWriter writer, IJsonConvertable val)
            {
                var serializer = new JsonSerializer();

                writer.WriteStartObject();

                writer.WritePropertyName(classPropertyName);
                writer.WriteValue(val.GetType().Name);

                writer.WritePropertyName(contentName);
                writer.WriteStartObject();
                foreach (var prop in val.GetJsonProperties())
                {
                    var type = prop.GetType();
                    writer.WritePropertyName(prop.Item1);
                    serializer.Serialize(writer, prop.Item2);
                }
                writer.WriteEndObject();

                writer.WriteEndObject();
            }

            public static object LoadJson(string json, Dictionary<string, Type> typeDict)
            {
                var root = JObject.Parse(json);
                var type = typeDict[root[classPropertyName].ToString()];

                return JsonConvert.DeserializeObject(root[contentName].ToString(), type);
            }

            public static Dictionary<string, Type> GetChildrenTypeOf<T>()
            {
                var types = Assembly.GetExecutingAssembly().GetTypes()
                    .Where(t => t.IsClass && !t.IsAbstract && typeof(T).IsAssignableFrom(t));

                var rs = new Dictionary<string, Type>();
                foreach (var t in types)
                    rs.Add(t.Name, t);

                return rs;
            }

            public static int DrawProbs(float[] probs)
            {
                float sum = 0;
                for (int i = 0; i < probs.Length; i++)
                    sum += probs[i];

                float rand = (float)new Random().NextDouble() * sum;
                for (int i = 0; i < probs.Length; i++)
                {
                    rand -= probs[i];
                    if (rand <= 1e-3)
                        return i;
                }

                return -1;
            }

            public static string ArrayToString(float[] floatArray, string format = "F2")
            {
                var sb = new StringBuilder();

                for (int i = 0; i < floatArray.Length; i++)
                {
                    sb.Append(floatArray[i].ToString(format));
                    if (i < floatArray.Length - 1)
                    {
                        sb.Append(", ");
                    }
                }

                return sb.ToString();
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