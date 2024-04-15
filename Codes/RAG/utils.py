import os
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core import load_index_from_storage


def build_sentence_window_index(
        documents,
        llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        sentence_window_size=3,
        save_dir="sentence_index",
):
    # Create the sentence window node parser with default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        # embed_model="local:BAAI/bge-large-en-v1.5",
        node_parser=node_parser
    )

    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context
        )

    return sentence_index


def get_sentence_window_query_engine(
        sentence_index,
        similarity_top_k=6,
        rerank_top=2
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

    # define reranker
    rerank = SentenceTransformerRerank(
        top_n=rerank_top,
        model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )

    return sentence_window_engine


######################################################################################################################
from trulens_eval import Feedback
from trulens_eval import OpenAI as fOpenAI
from trulens_eval import Feedback
from trulens_eval import TruLlama
from trulens_eval.feedback import Groundedness
import numpy as np

provider = fOpenAI()
f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer relevance"
).on_input().on_output()

context_selection = TruLlama.select_source_nodes().node.text

f_qs_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)

grounded = Groundedness(groundedness_provider=provider)
f_groundedness = (
    Feedback(
        grounded.groundedness_measure_with_cot_reasons,
        name="Groundedness",
    )
    .on(context_selection)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)


def get_prebuilt_trulens_recorder(sentence_window_engine, app_id):
    tru_recorder = TruLlama(
        app=sentence_window_engine,  ## LLamaindex app
        app_id=app_id,
        feedbacks=[
            f_qa_relevance,
            f_qs_relevance,
            f_groundedness
        ]
    )
    return tru_recorder
