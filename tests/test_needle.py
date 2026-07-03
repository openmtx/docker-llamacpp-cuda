import sys

import requests

from config import build_args, resolve
from utils import RESULTS, log_result, print_summary, reset_results, chat


WIKIPEDIA = """Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals.

High-profile applications of AI include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); virtual assistants (e.g., Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (e.g., language models and AI art); and superhuman play and analysis in strategy games (e.g., chess and Go). However, many AI applications are not perceived as such: "A lot of cutting-edge AI has filtered into general applications, often without being called AI because once something becomes useful enough and common enough it's not labeled AI anymore."

Various subfields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include learning, reasoning, knowledge representation, planning, natural language processing, and perception, as well as support for robotics. To reach these goals, AI researchers have adapted and integrated a wide range of techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics. AI also draws upon psychology, linguistics, philosophy, neuroscience, and other fields.

The general problem of simulating (or creating) intelligence has been broken into subproblems. These consist of particular traits or capabilities that researchers expect an intelligent system to display.

Reasoning and problem-solving: Early researchers developed algorithms that imitated step-by-step reasoning that humans use when they solve puzzles or make logical deductions. By the late 1980s and 1990s, methods were developed for dealing with uncertain or incomplete information, employing concepts from probability and economics. Many of these algorithms are insufficient for solving large reasoning problems because they experience a "combinatorial explosion": They become exponentially slower as the problems grow.

Knowledge representation: Knowledge representation and knowledge engineering allow AI programs to answer questions intelligently and make deductions about real-world facts. Formal knowledge representations are used in content-based indexing and retrieval, scene interpretation, clinical decision support, knowledge discovery, and other areas. Among the most difficult problems in knowledge representation are the breadth of commonsense knowledge (the set of atomic facts that the average person knows is enormous); and the sub-symbolic form of most commonsense knowledge (much of what people know is not represented as "facts" or "statements" that they could express verbally).

Planning and decision-making: An "agent" is any entity (artificial or not) that perceives and takes actions in the world. A rational agent has goals or preferences and takes actions to make them happen. In automated planning, the agent has a specific goal. In automated decision-making, the agent has preferences—there are some situations it would prefer to be in, and some situations it is trying to avoid.

Machine learning is the study of programs that can improve their performance on a given task automatically. It has been a part of AI from the beginning. There are several kinds of machine learning. Unsupervised learning analyzes a stream of data and finds patterns and makes predictions without any other guidance. Supervised learning requires labeling the training data with the expected answers, and comes in two main varieties: classification (where the program must learn to predict what category the input belongs in) and regression (where the program must deduce a numeric function based on numeric input). In reinforcement learning, the agent is rewarded for good responses and punished for bad ones.

Natural language processing (NLP) allows programs to read, write and communicate in human languages. Specific problems include speech recognition, speech synthesis, machine translation, information extraction, information retrieval and question answering. Modern deep learning techniques for NLP include word embedding (representing words, typically as vectors encoding their meaning), transformers (a deep learning architecture using an attention mechanism), and others.

Machine perception is the ability to use input from sensors (such as cameras, microphones, wireless signals, active lidar, sonar, radar, and tactile sensors) to deduce aspects of the world. Computer vision is the ability to analyze visual input. The field includes speech recognition, image classification, facial recognition, object recognition, object tracking, and robotic perception.

Affective computing is a field that comprises systems that recognize, interpret, process, or simulate human feeling, emotion, and mood. For example, some virtual assistants are programmed to speak conversationally or even to banter humorously; it makes them appear more sensitive to the emotional dynamics of human interaction, or to otherwise facilitate human–computer interaction.

AI research uses a wide variety of techniques to accomplish the goals above. Search and optimization include state space search and local search. State space search searches through a tree of possible states to try to find a goal state. Local search uses mathematical optimization to find a solution to a problem. Gradient descent is a type of local search that optimizes a set of numerical parameters by incrementally adjusting them to minimize a loss function.

Formal logic is used for reasoning and knowledge representation. Formal logic comes in two main forms: propositional logic (which operates on statements that are true or false and uses logical connectives such as "and", "or", "not" and "implies") and predicate logic (which also operates on objects, predicates and relations and uses quantifiers such as "Every X is a Y" and "There are some Xs that are Ys").

Many problems in AI (including reasoning, planning, learning, perception, and robotics) require the agent to operate with incomplete or uncertain information. AI researchers have devised a number of tools to solve these problems using methods from probability theory and economics.

An artificial neural network is based on a collection of nodes also known as artificial neurons, which loosely model the neurons in a biological brain. It is trained to recognise patterns; once trained, it can recognise those patterns in fresh data. There is an input, at least one hidden layer of nodes and an output. Each node applies a function and once the weight crosses its specified threshold, the data is transmitted to the next layer.

Deep learning is a subset of machine learning, which is itself a subset of artificial intelligence. Deep learning uses several layers of neurons between the network's inputs and outputs. The multiple layers can progressively extract higher-level features from the raw input. Deep learning has profoundly improved the performance of programs in many important subfields of artificial intelligence, including computer vision, speech recognition, natural language processing, image classification, and others.

Generative pre-trained transformers (GPT) are large language models (LLMs) that generate text based on the semantic relationships between words in sentences. Text-based GPT models are pre-trained on a large corpus of text that can be from the Internet. The pretraining consists of predicting the next token. Throughout this pretraining, GPT models accumulate knowledge about the world and can then generate human-like text by repeatedly predicting the next token.

AI and machine learning technology is used in most of the essential applications of the 2020s, including: search engines (such as Google Search), targeting online advertisements, recommendation systems (offered by Netflix, YouTube or Amazon), virtual assistants (such as Siri or Alexa), autonomous vehicles (including drones, ADAS and self-driving cars), automatic language translation, facial recognition, and image labeling.

AlphaFold 2 demonstrated the ability to approximate, in hours rather than months, the 3D structure of a protein. In 2023, it was reported that AI-guided drug discovery helped find a class of antibiotics capable of killing two different types of drug-resistant bacteria.

Game playing programs have been used since the 1950s to demonstrate and test AI's most advanced techniques. Deep Blue became the first computer chess-playing system to beat a reigning world chess champion, Garry Kasparov, on 11 May 1997. In 2011, in a Jeopardy! quiz show exhibition match, IBM's question answering system, Watson, defeated the two greatest Jeopardy! champions. In March 2016, AlphaGo won 4 out of 5 games of Go in a match with Go champion Lee Sedol.

Various countries are deploying AI military applications. The main applications enhance command and control, communications, sensors, integration and interoperability. Research is targeting intelligence collection and analysis, logistics, cyber operations, information operations, and semiautonomous and autonomous vehicles.

Generative artificial intelligence, also known as generative AI or GenAI, is a subfield of artificial intelligence that uses generative models to generate text, images, videos, audio, software code or other forms of data. These models learn the underlying patterns and structures of their training data, and use them to generate new data in response to input.

AI has potential benefits and potential risks. AI may be able to advance science and find solutions for serious problems. However, as the use of AI has become widespread, several unintended consequences and risks have been identified. In-production systems can sometimes not factor ethics and bias into their AI training processes, especially when the AI algorithms are inherently unexplainable in deep learning.

Machine learning algorithms require large amounts of data. The techniques used to acquire this data have raised concerns about privacy, surveillance and copyright. AI-powered devices and services, such as virtual assistants and IoT products, continuously collect personal information, raising concerns about intrusive data gathering and unauthorized access by third parties.

The commercial AI scene is dominated by Big Tech companies such as Alphabet Inc., Amazon, Apple Inc., Meta Platforms, and Microsoft. Some of these players already own the vast majority of existing cloud infrastructure and computing power from data centers, allowing them to entrench further in the marketplace.

Fueled by a growth in artificial intelligence, data centers' demand for power increased in the 2020s. In January 2024, the International Energy Agency released Electricity 2024, forecasting that power demand for data centers and AI might double by 2026, with additional electric power usage equal to electricity used by the whole Japanese nation.

Artificial intelligence was founded as an academic discipline in 1956, and the field went through multiple cycles of optimism throughout its history, followed by periods of disappointment and loss of funding, known as AI winters. Funding and interest increased substantially after 2012, when graphics processing units began being used to accelerate neural networks, and deep learning outperformed previous AI techniques. This growth accelerated further after 2017 with the transformer architecture. In the 2020s, an ongoing period of rapid progress in advanced generative AI became known as the AI boom.

Philosophical questions about AI include: Can a machine be intelligent? Can it be conscious? Can it feel? These questions have been discussed by scientists, philosophers, and writers for decades. Some argue that AI will never truly be intelligent because it is just a tool, while others believe that AI could eventually surpass human intelligence.

The future of AI is uncertain, but most experts believe that AI will continue to advance and become more capable. Some worry about the potential for AI to be used for harmful purposes, while others are optimistic about the potential benefits. The development of artificial general intelligence (AGI) - AI that can match or exceed human intelligence across a wide range of tasks - remains a long-term goal for many researchers."""


def generate_haystack(target_tokens):
    chars_needed = int(target_tokens * 4)
    repeats = (chars_needed // len(WIKIPEDIA)) + 1
    content = WIKIPEDIA * repeats
    return content[:chars_needed]


def test_needle(chat_url, headers, model, timeout):
    print("\n=== Needle-in-Haystack (long-context recall) ===\n")
    secret = "THE_SECRET_CODE_IS_QUADRO_POWER_9000"
    sizes = [16000, 80000]
    for target_ctx in sizes:
        part_1 = generate_haystack(int(target_ctx * 0.85))
        part_2 = generate_haystack(int(target_ctx * 0.15))
        full = f"""{part_1}

--- Research Notes ---
Document ID: 2024-AI-001
Topic: AI Systems and Security
Note: Various AI systems require unique authentication codes for access.
{secret}
This code is updated quarterly and should be kept confidential.
Reviewed by: Systems Team
---

{part_2}"""
        print(f"  {target_ctx:,} token context... ", end="", flush=True)
        try:
            resp, _, dt, usage = chat(
                chat_url, headers, model,
                [
                    {"role": "system",
                     "content": "You are a helpful assistant that carefully "
                                "reads documents and finds specific information."},
                    {"role": "user",
                     "content": f"LOGS:\n{full}\n\nWhat is the secret code?"},
                ],
                temperature=0.0, max_tokens=8192, timeout=timeout,
            )
        except Exception as e:
            print(f"FAIL: {e}")
            log_result(f"needle_{target_ctx}", False, str(e))
            continue
        found = "QUADRO_POWER_9000" in resp.upper()
        pt = usage.get("prompt_tokens", 0)
        print(f"{'PASS' if found else 'FAIL'} ({dt:.1f}s, {pt:,} tok)")
        log_result(f"needle_{target_ctx}", found, resp[:100])


def main(args=None):
    ap = build_args("Needle-in-haystack recall test")
    if args is None:
        args = ap.parse_args()
    chat_url, _, headers, model, timeout = resolve(args)
    try:
        requests.get(f"http://{args.host}:{args.port}/health",
                     timeout=5).raise_for_status()
    except Exception as e:
        print(f"Server not reachable: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Server: {args.host}:{args.port}  |  Model: {model}")
    reset_results()
    test_needle(chat_url, headers, model, timeout)
    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
