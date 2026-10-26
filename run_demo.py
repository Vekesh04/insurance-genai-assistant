from graph.workflow import ask_assistant

questions = [
    "What is the status of claim CLM300001?",
    "What is the maternity waiting period under Health Secure Plus?",
    "Analyze claim CLM300001 and tell me if documents are missing.",
]

for question in questions:
    result = ask_assistant(question)
    print("\nQUESTION:", question)
    print("ROUTE:", result["route"])
    print("ANSWER:", result["answer"])
