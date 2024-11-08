from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from artifacts.pdf import Pdf
from config import ConfigManager
from utils.validator import validate_args
from dotenv import load_dotenv


PROMPT_TEMPLATE = """
You are tasked with extracting the geographical breakdown of revenue from a large text corpus. Your goal is to identify and organize information about revenue distribution across different geographical regions or countries.

First, carefully read through the following text corpus:

<text_corpus>
{TEXT_CORPUS}
</text_corpus>

To extract the geographical breakdown of revenue, follow these steps:

1. Analyze the text corpus, paying close attention to any mentions of revenue, sales, or financial performance associated with specific countries, regions, or geographical areas.

2. Look for keywords such as "revenue," "sales," "earnings," "market," or "performance" in proximity to geographical names or regions.

3. Identify specific monetary values or percentages associated with each geographical location mentioned in relation to revenue or sales.

4. Create a list of all geographical entities (countries, regions, continents) mentioned in connection with revenue data.

5. For each geographical entity, note down the corresponding revenue figure, whether it's an absolute value or a percentage. If both are provided, include both.

6. If the text mentions time periods (e.g., fiscal years, quarters) for the revenue data, include this information as well.

7. In cases where the geographical breakdown is given in terms of percentages without specific monetary values, calculate the monetary values if the total revenue is provided.

8. If you encounter any unclear or ambiguous information, make a note of it and provide your best interpretation based on the context.

9. Organize the extracted information in a structured format, grouping related geographical entities if appropriate (e.g., grouping countries under their respective continents or regions).

Present your findings in the following format:

<geographical_revenue_breakdown>
<region name="[Region/Country Name]">
<revenue>[Monetary Value and/or Percentage]</revenue>
<time_period>[Time Period, if specified]</time_period>
<notes>[Any additional relevant information or clarifications]</notes>
</region>
<!-- Repeat for each geographical entity -->
<additional_information>
[Include any overall observations, totals, or other relevant details not fitting into the above structure]
</additional_information>
</geographical_revenue_breakdown>

If you cannot find any relevant information about geographical revenue breakdown in the provided text corpus, state this clearly in your response.

Remember to base your extraction solely on the information provided in the text corpus. Do not include any external knowledge or assumptions not supported by the given text.

"""

class TxtExtractor:

    @validate_args({"txt": str})
    def __init__(self, txt: str):
        load_dotenv()
        self.model = ChatOpenAI(model=ConfigManager().openai_llm)
        self.txt = txt
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = self.prompt | self.model | StrOutputParser()

    def extract(self):
        return self.chain.invoke({"TEXT_CORPUS": self.txt})


if __name__ == "__main__":
    try:
        pdf = Pdf(ConfigManager().pdf_file_path)
        text = pdf.extract_text()
        extractor = TxtExtractor(text)
        extraction_result = extractor.extract()
        print(extraction_result)
    except Exception as e:
        print(f"Error: {str(e)}")
