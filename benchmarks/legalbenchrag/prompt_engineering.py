import asyncio
import os
from pathlib import Path

from sac_rag.utils.credentials import credentials
from sac_rag.utils.ai import generate_document_summary
from sac_rag.data_models import Document
from sac_rag.utils.ai import AIModel


def read_file(file_path) -> str:
    """Reads the prompt template from a file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


doc_dir = Path.cwd() / "data" / "corpus"
# doc1_path = doc_dir / "contractnli" / "ceii-and-nda.txt"
# doc1_path = doc_dir / "contractnli" / "RROI_Confidentiality_Agreement_Final.txt"
# doc1_path = doc_dir / "contractnli" / "SE_NDCA_and_PRE-QUAL_PACKAGE_March-2016.txt"
# doc1_path = doc_dir / "privacy_qa" / "Wordscapes.txt"
# doc1_path = doc_dir / "privacy_qa" / "Keep.txt"
# doc1_path = doc_dir / "cuad" / "VISIUMTECHNOLOGIES,INC_10_20_2004-EX-10.20-DISTRIBUTOR AGREEMENT.txt"
# doc1_path = doc_dir / "cuad" / "OAKTREECAPITALGROUP,LLC_03_02_2020-EX-10.8-Services Agreement.txt"
# doc1_path = doc_dir / "maud" / "TIFFANY_&_CO._LVMH_MOËT_HENNESSY-LOUIS_VUITTON.txt"
doc1_path = doc_dir / "maud" / "Community Bankers Trust Corporation_United Bankshares, Inc..txt"
documents = [
    Document(file_path=str(doc1_path), content=read_file(doc1_path)),
]
# prompt_file_path = Path.cwd() / "data" / "prompt-sumV1.txt"
prompt_file_path = Path.cwd() / "data" / "prompt-sumV2.txt"
summary_prompt_template = read_file(prompt_file_path)
# print(summary_prompt_template[:1000])


async def main():
    os.environ["OPENAI_API_KEY"] = credentials.ai.openai_api_key.get_secret_value()
    os.environ["COHERE_API_KEY"] = credentials.ai.cohere_api_key.get_secret_value()
    os.environ["VOYAGEAI_API_KEY"] = credentials.ai.voyageai_api_key.get_secret_value()

    summarization_model = AIModel(company="openai", model="gpt-4o-mini")

    document_summaries = {}
    for doc in documents:
        document_summary = await generate_document_summary(
            document_file_path=doc.file_path,
            document_content=doc.content,
            summarization_model=summarization_model,
            summary_prompt_template=summary_prompt_template,
            prompt_target_char_length=300,
            truncate_char_length=350,
            use_cache=False,
            summaries_output_dir_base=Path.cwd() / "data" / "summaries"
        )
        document_summaries[doc.file_path] = document_summary

    print(document_summaries)


if __name__ == "__main__":
    asyncio.run(main())
