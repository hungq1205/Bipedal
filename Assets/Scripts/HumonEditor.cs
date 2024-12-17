using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(Humon))]
public class HumonEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var humon = (Humon)target;

        if (GUILayout.Button("Kill", GUILayout.Height(30)))
            humon.Conclude(Lib.AI.RL.ConcludeType.Killed);

        if (GUILayout.Button("Set Exploration", GUILayout.Height(30)))
            humon.exploration.exploreRate = humon.explorationRate;

        GUILayout.BeginHorizontal();

        if (GUILayout.Button("Save", GUILayout.Height(30)))
            humon.Policy.Save(humon.saveFilePath + humon.saveFilename);
        if (GUILayout.Button("Load", GUILayout.Height(30)))
            humon.Policy.Load(humon.saveFilePath + humon.saveFilename);

        GUILayout.EndHorizontal();
    }
}
