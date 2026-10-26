from graph.router import route_question


def test_sql_route():
    assert route_question("What is the status of claim CLM300001?").route == "sql"


def test_rag_route():
    assert route_question("What is the maternity waiting period under Health Secure Plus?").route == "rag"


def test_combined_route():
    assert route_question("Analyze claim CLM300001 and tell me if documents are missing.").route == "combined"
