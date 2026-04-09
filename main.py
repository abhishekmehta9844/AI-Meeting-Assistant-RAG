from rag_pipeline import build_rag


def main():
    print("Hello from data-injestion-pipeline!")


if __name__ == "__main__":
    rag = build_rag()
    print(rag.query("what does the company do"))
