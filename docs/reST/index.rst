using UnityEngine;
using UnityEngine.UI;

public class FootballGame : MonoBehaviour
{
    [Header("References")]
    public GameObject player;
    public GameObject ai;
    public GameObject ball;
    public Transform playerGoal;
    public Transform aiGoal;
    public Text scoreText;

    [Header("Settings")]
    public float moveSpeed = 6f;
    public float rotateSpeed = 120f;
    public float kickForce = 12f;
    public float aiSpeed = 4f;

    private Rigidbody playerRb;
    private Rigidbody aiRb;
    private Rigidbody ballRb;

    private int playerScore = 0;
    private int aiScore = 0;

    void Start()
    {
        playerRb = player.GetComponent<Rigidbody>();
        aiRb = ai.GetComponent<Rigidbody>();
        ballRb = ball.GetComponent<Rigidbody>();

        UpdateScore();
    }

    void Update()
    {
        PlayerRotate();
        AIControl();
    }

    void FixedUpdate()
    {
        PlayerMove();
    }

    // ================= PLAYER =================

    void PlayerMove()
    {
        float move = Input.GetAxis("Vertical");
        Vector3 moveDir = player.transform.forward * move * moveSpeed;
        playerRb.velocity = new Vector3(moveDir.x, playerRb.velocity.y, moveDir.z);
    }

    void PlayerRotate()
    {
        float rotate = Input.GetAxis("Horizontal");
        player.transform.Rotate(Vector3.up * rotate * rotateSpeed * Time.deltaTime);

        if (Input.GetKeyDown(KeyCode.Space))
        {
            TryKick(player.transform);
        }
    }

    // ================= AI =================

    void AIControl()
    {
        float distance = Vector3.Distance(ai.transform.position, ball.transform.position);

        if (distance < 20f)
        {
            Vector3 direction = (ball.transform.position - ai.transform.position).normalized;

            aiRb.velocity = new Vector3(direction.x * aiSpeed, aiRb.velocity.y, direction.z * aiSpeed);

            ai.transform.LookAt(ball.transform);

            if (distance < 2f)
            {
                TryKick(ai.transform, kickForce - 3);
            }
        }
    }

    // ================= KICK =================

    void TryKick(Transform kicker, float force = -1)
    {
        float distance = Vector3.Distance(kicker.position, ball.transform.position);

        if (distance < 2f)
        {
            Vector3 direction = (ball.transform.position - kicker.position).normalized;

            float finalForce = (force == -1) ? kickForce : force;

            ballRb.AddForce(direction * finalForce, ForceMode.Impulse);
        }
    }

    // ================= GOAL DETECTION =================

    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject == ball)
        {
            if (Vector3.Distance(ball.transform.position, playerGoal.position) < 3f)
            {
                aiScore++;
            }
            else if (Vector3.Distance(ball.transform.position, aiGoal.position) < 3f)
            {
                playerScore++;
            }

            UpdateScore();
            ResetPositions();
        }
    }

    // ================= RESET =================

    void ResetPositions()
    {
        ball.transform.position = new Vector3(0, 1, 0);
        ballRb.velocity = Vector3.zero;
        ballRb.angularVelocity = Vector3.zero;

        player.transform.position = new Vector3(0, 1, -10);
        ai.transform.position = new Vector3(0, 1, 10);

        playerRb.velocity = Vector3.zero;
        aiRb.velocity = Vector3.zero;
    }

    void UpdateScore()
    {
        if (scoreText != null)
            scoreText.text = "Player: " + playerScore + "  |  AI: " + aiScore;
    }
}
