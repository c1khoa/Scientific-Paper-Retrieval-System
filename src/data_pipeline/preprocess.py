import pandas as pd
import re

class PaperPreprocessor:

    def clean_arxiv_id(self, url):
        match = re.search(r'(\d{4}\.\d+)', str(url))
        return match.group(1) if match else url

    def clean_text(self, text):

        if pd.isna(text):
            return ""

        text = str(text)

        text = re.sub(r'\\cite\{.*?\}', '', text)
        text = re.sub(r'\\ref\{.*?\}', '', text)
        text = re.sub(r'\\url\{.*?\}', '', text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def build_doc_text(self, title, abstract):

        title = self.clean_text(title)
        abstract = self.clean_text(abstract)

        text = f"{title}. {title}. {abstract}"

        return text.strip()
    
    def preprocess_dataset(self, path, output_path):

        df = pd.read_parquet(path)

        df = df.rename(columns={"id": "doc_id"})

        df["doc_id"] = df["doc_id"].apply(self.clean_arxiv_id)

        title = df["title"].fillna("").astype(str)
        abstract = df["abstract"].fillna("").astype(str)

        title = title.str.replace(r'\\cite\{.*?\}', '', regex=True)
        title = title.str.replace(r'\\ref\{.*?\}', '', regex=True)
        title = title.str.replace(r'\\url\{.*?\}', '', regex=True)
        title = title.str.replace(r'\s+', ' ', regex=True)

        abstract = abstract.str.replace(r'\\cite\{.*?\}', '', regex=True)
        abstract = abstract.str.replace(r'\\ref\{.*?\}', '', regex=True)
        abstract = abstract.str.replace(r'\\url\{.*?\}', '', regex=True)
        abstract = abstract.str.replace(r'\s+', ' ', regex=True)

        df["text"] = title + ". " + title + ". " + abstract

        df = df[["doc_id", "text"]]
        df = df.drop_duplicates(subset="doc_id")
        
        df.to_parquet(output_path, index=None)

        return df