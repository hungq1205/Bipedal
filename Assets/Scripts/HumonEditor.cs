using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(Humon))]
public class HumonEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var humon = (Humon)target;

        GUILayout.BeginHorizontal();

        if (GUILayout.Button("Save", GUILayout.Height(30)))
            humon.Policy.Save(humon.saveFilepath);
        if (GUILayout.Button("Load", GUILayout.Height(30)))
            humon.Policy.Load(humon.saveFilepath);

        GUILayout.EndHorizontal();
    }
}
