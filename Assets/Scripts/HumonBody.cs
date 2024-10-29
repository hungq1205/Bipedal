using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HumonBody : MonoBehaviour
{
    public event Action<Humon> onLyingGround;

    bool isLyingGround;

    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.transform.CompareTag("Ground"))
        {
            isLyingGround = true;
            StartCoroutine(GroundTrigger(0));
        }
    }

    private void OnCollisionExit2D(Collision2D collision)
    {
        if (collision.transform.CompareTag("Ground"))
        {
            isLyingGround = false;
            StopAllCoroutines();
        }
    }

    IEnumerator GroundTrigger(int miliSec = 2000)
    {
        yield return new WaitForSeconds(miliSec / 1000);
        if (isLyingGround)
            onLyingGround?.Invoke(transform.parent.GetComponent<Humon>());
    }
}
