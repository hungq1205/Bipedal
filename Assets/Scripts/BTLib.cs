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

                Layers = bundle.Item1;
                Weights = bundle.Item2;
                Optimizer = new SGD(learningRate);
                InDim = Layers[0].dim;
                OutDim = Layers[Layers.LongLength - 1].dim;

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

                Layers = bundle.Item1;
                Weights = bundle.Item2;
                Optimizer = optimizer;
                InDim = Layers[0].dim;
                OutDim = Layers[Layers.LongLength - 1].dim;

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
                        weights[i - 1] = layers[i].GenerateWeightMatrix();

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

        public class SGD : Optimizer
        {
            public float learningRate;

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

            public override IEnumerable<(string, object)> GetJsonProperties()
            {
                yield return ("learningRate", learningRate);
                yield return ("weightDecay", weightDecay);
            }
        }

        #endregion

        #region Layer

        public enum ActivationFunc
        {
            Tanh = 2,
            Linear = 5,
            Softmax = 6
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
                    case ActivationFunc.Tanh:
                        float sqrExp = MathF.Exp(x) * loss;
                        sqrExp *= sqrExp;
                        return 4 * loss / (sqrExp + (1 / sqrExp) + 2);
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
                    case ActivationFunc.Tanh:
                        for (int i = 0; i < X.Length; i++)
                        {
                            float sqrExp = MathF.Exp(X[i]);
                            sqrExp *= sqrExp;
                            result[i] = 4 * loss[i] / (sqrExp + (1 / sqrExp) + 2);
                        }
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
                    case ActivationFunc.Tanh:
                        return MathF.Tanh(x);
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
                    case ActivationFunc.Tanh:
                        for (int i = 0; i < X.Length; i++)
                            result[i] = MathF.Tanh(X[i]);
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
                        // Debug
                        if (Humon.PrintPred)
                            UnityEngine.Debug.Log(string.Join(", ", X.Select(f => f.ToString("F2"))) + "\n" + string.Join(", ", result.Select(f => f.ToString("F2"))));
                        break;
                    case ActivationFunc.Linear:
                    default:
                        return X;
                }

                return result;
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

                float[] GetPred(float[] obs);

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

                public virtual float[] GetPred(float[] obs) => Policy.Forward(obs);

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

                public virtual float[] GetPred(float[] obs) => Policy.Forward(obs);

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

                public virtual float[] GetPred(float[] obs) => Policy.Forward(obs);

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
    }
}