import os
import json
import pandas as pd
from typing import List, Dict
from ast import literal_eval
from src.utils import _get_tags_dict
from collections import defaultdict
from tqdm import tqdm
from data_generation import generate_one_llm_input, postprocess_RAG_answers
from llm_multiprocessing_inference import get_answers

executive_summary_prompt = """
I want you to act like a data analyst.
I will provide you with a JSON dict where the keys are the breakdowns and the values are the breakdown analyses.
You will need to generate an executive summary and a comparison of the analyses generated for the different breakdowns.
The values discuss about the following indicator: %s.

The output is a markdown text with bullet points format:
"""


breakdowns_metadata = {
    "No Breakdown": [],
    # "Interviewee Location Type",
    # "Interviewee Organization Type",
    "Interviewee Gender": ["M", "F"],
}

df_relevant_columns = [
    "entry_id",
    "Entry Type",
    "Extracted Entries",
    "Document Publishing Date",
    "Document Title",
    "Document Source",
    "entry_fig_path",
]


def _custom_eval(x):
    try:
        return literal_eval(x)
    except:
        return x


def _load_classification_dataset(
    classification_dataset_path: os.PathLike,
    eval_cols: List[str] = [
        "Interviewee",
        # "Secondary Tags", # not used for now
        "Framework Probabilities",
    ],
) -> pd.DataFrame:
    df = pd.read_csv(classification_dataset_path)
    for col in eval_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataset")
        df[col] = df[col].apply(_custom_eval)

    df["Interviewee Location Type"] = df["Interviewee"].apply(
        lambda x: ", ".join(
            sorted(
                list(set([i["location_type"] for i in x if i["location_type"] != "-"]))
            )
        ) if type(x) == list else "-"
    )
    df["Interviewee Organization Type"] = df["Interviewee"].apply(
        lambda x: ", ".join(
            sorted(
                list(
                    set(
                        [
                            i["organisation_type"]
                            for i in x
                            if i["organisation_type"] != "-"
                        ]
                    )
                ) if type(x) == list else "-"
            )
        )
    )
    df["Interviewee Gender"] = df["Interviewee"].apply(
        lambda x: ", ".join(
            sorted(list(set([i["gender"] for i in x if i["gender"] != "-"])))
        ) if type(x) == list else "-"
    )

    return df


def _generate_no_breakdown_analysis(
    indicator: str, df_one_tag: pd.DataFrame, tqdm_bar: tqdm
):
    analyses = defaultdict(dict)
    analyses["By Breakdown"] = {}

    df_one_tag["question_id"] = 0
    # print(df_one_tag.columns)
    prompt = generate_one_llm_input(
        df_one_tag, 5, indicator, "Extracted Entries", "", "english"
    )
    answers = get_answers(
        [prompt],
        api_pipeline="Ollama",
        model="deepseek-r1:14b-qwen-distill-q4_K_M",
        response_type="structured",
        default_response="{}",
        show_progress=False,
    )
    # print(answers)
    final_answer = postprocess_RAG_answers(answers, df_one_tag, df_relevant_columns)[0]
    # print(final_answer)
    analyses["Executive Summary"] = final_answer
    tqdm_bar.update(1)

    return analyses

def _generate_executive_summary(input_analyses: Dict[str, str], indicator: str):
    system_prompt = executive_summary_prompt % indicator
    user_prompt = json.dumps(input_analyses)
    prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    answers = get_answers(
        [prompt],
        api_pipeline="Ollama",
        model="deepseek-r1:14b-qwen-distill-q4_K_M",
        response_type="unstructured",
        default_response="",
        show_progress=False,
    )
    return answers[0]


def _generate_breakdown_analysis(
    indicator: str,
    df_one_tag: pd.DataFrame,
    breakdown: str,
    breakdown_values: List[str],
    tqdm_bar: tqdm,
):
    analyses = defaultdict(dict)
    done_breakdown_analyses = {}
    for breakdown_value in breakdown_values:
        df_one_breakdown = df_one_tag.copy()
        df_one_breakdown = df_one_breakdown[
            df_one_breakdown[breakdown] == breakdown_value
        ]
        if len(df_one_breakdown) == 0:
            tqdm_bar.update(1)
            continue
        
        summary_one_breakdown = _generate_no_breakdown_analysis(
            indicator, df_one_breakdown, tqdm_bar
        )
        analyses["By Breakdown"][breakdown_value] = summary_one_breakdown
        done_breakdown_analyses[breakdown_value] = summary_one_breakdown["final_answer"]

    if len(done_breakdown_analyses) > 1:
        executive_summary = _generate_executive_summary(done_breakdown_analyses, indicator)
    else:
        executive_summary = ""
        
    analyses["Executive Summary"] = executive_summary
    tqdm_bar.update(1)

    return analyses


def generate_analyses(
    classification_dataset_path: os.PathLike,
    output_path: os.PathLike,
    breakdown_metadata: Dict[str, List[str]] = breakdowns_metadata,
):
    df = _load_classification_dataset(classification_dataset_path)
    classification_tags = sorted(list(df["Framework Probabilities"].iloc[0].keys()))
    tags_dict = _get_tags_dict(classification_tags)

    output_dict = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
    )
    project_subproject_dict = defaultdict(list)
    tot_n_subprojects = 0
    projects = df["project_name"].unique()
    for project in projects:
        project_subproject_dict[project] = df[df["project_name"] == project][
            "sub_project_name"
        ].unique()
        tot_n_subprojects += len(project_subproject_dict[project])
    n_breakdown_metadata_generation_per_tag = 1 + 3 * (len(breakdown_metadata) - 1)
    n_llm_calls = (
        tot_n_subprojects
        * len(classification_tags)
        * n_breakdown_metadata_generation_per_tag
    )

    tqdm_bar = tqdm(tags_dict.items(), total=n_llm_calls, desc="Generating analyses")
    for project, subprojects in project_subproject_dict.items():
        for subproject in subprojects:
            for pillar, subpillars in tags_dict.items():
                for subpillar, indicators in subpillars.items():
                    for indicator in indicators:
                        df_one_tag = df.copy()
                        tag = f"{pillar}->{subpillar}->{indicator}"
                        df_one_tag = df_one_tag[
                            (df_one_tag["project_name"] == project)
                            & (df_one_tag["sub_project_name"] == subproject)
                        ]
                        df_one_tag["Framework Probabilities"] = df_one_tag[
                            "Framework Probabilities"
                        ].apply(lambda x: x[tag])
                        df_one_tag = df_one_tag[
                            df_one_tag["Framework Probabilities"] >= 0.6
                        ]
                        df_one_tag.sort_values(
                            by="Framework Probabilities", ascending=False, inplace=True
                        )

                        for breakdown, breakdown_values in breakdown_metadata.items():
                            if breakdown == "No Breakdown":
                                analyses = _generate_no_breakdown_analysis(
                                    indicator, df_one_tag, tqdm_bar
                                )
                            else:
                                analyses = _generate_breakdown_analysis(
                                    indicator,
                                    df_one_tag,
                                    breakdown,
                                    breakdown_values,
                                    tqdm_bar,
                                )

                            output_dict[project][subproject][pillar][subpillar][
                                indicator
                            ][breakdown] = analyses

                            with open(output_path, "w") as f:
                                json.dump(output_dict, f)
