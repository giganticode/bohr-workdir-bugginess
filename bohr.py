from typing import Dict, Optional, List

from bohrapi.artifacts import Commit
from bohrlabels.labels import CommitLabel
from bohrapi.core import Dataset, Task, Workspace, Experiment


def id_with_files(id: str, conditions: Optional[List[Dict]] = None) -> Dict:
    return {"$and": [
        {id: {"$exists": True}},
        {"files": {"$exists": True}},
        {"files": {"$type": "array"}}] + (conditions if conditions is not None else [])}


commits_200k_files = Dataset(id='commits_200k_files', top_artifact=Commit, query=id_with_files('bohr.200k_commits'))
commits_200k_files_no_merges = Dataset(id='commits_200k_files_no_merges', top_artifact=Commit, query=id_with_files("bohr.200k_commits", [{"special_commit_finder/0_1.merge": False}]))

berger_files = Dataset(id='berger_files', top_artifact=Commit, query=id_with_files('manual_labels.berger'))
levin_files = Dataset(id='levin_files', top_artifact=Commit, query=id_with_files('manual_labels.levin'))
herzig = Dataset(id='manual_labels.herzig', top_artifact=Commit)
mauczka_files = Dataset(id='mauczka_files', top_artifact=Commit, query=id_with_files('manual_labels.mauczka'))
idan_files = Dataset(id='idan_files', top_artifact=Commit, query=id_with_files('idan/0_1', [{"idan/0_1.Is_Corrective": {"$exists": True}}]))

herzig_train = Dataset(id='bohr.herzig_train', top_artifact=Commit)
herzig_eval = Dataset(id='bohr.herzig_eval', top_artifact=Commit)

levin_small_changes = Dataset(id='levin_small_changes', top_artifact=Commit, query=id_with_files('manual_labels.levin', [{'bohr.gt_512_codeberta_tokens': False}]))
levin_large_changes = Dataset(id='levin_large_changes', top_artifact=Commit, query=id_with_files('manual_labels.levin', [{'bohr.gt_512_codeberta_tokens': True}]))

berger_small_changes = Dataset(id='berger_small_changes', top_artifact=Commit, query=id_with_files('manual_labels.berger', [{'bohr.gt_512_codeberta_tokens': False}]))
berger_large_changes = Dataset(id='berger_large_changes', top_artifact=Commit, query=id_with_files('manual_labels.berger', [{'bohr.gt_512_codeberta_tokens': True}]))

bohr_200k_small_changes = Dataset(id='bohr_200k_small_changes', top_artifact=Commit, query=id_with_files('bohr.200k_commits', [{'bohr.gt_512_codeberta_tokens': False}]))
bohr_200k_large_changes = Dataset(id='bohr_200k_large_changes', top_artifact=Commit, query=id_with_files('bohr.200k_commits', [{'bohr.gt_512_codeberta_tokens': True}]))

#conventional_with_files = Dataset(id='conventional_with_files', top_artifact=Commit, query={'conventional_commit/0_1.conventional': True, 'message': {"$exists": True}, 'files': {"$exists": True}}, n_datapoints=20000)


bugginess = Task(name='bugginess', author='hlib', description='bug or not', top_artifact=Commit,
                 labels=[CommitLabel.NonBugFix, CommitLabel.BugFix],
                 training_dataset=commits_200k_files_no_merges,
                 test_datasets={
                                commits_200k_files: None,
                                idan_files: lambda c: (CommitLabel.BugFix if c.raw_data['idan/0_1']['Is_Corrective'] else CommitLabel.NonBugFix),
                                levin_files: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['levin']['bug'] == 1 else CommitLabel.NonBugFix),
                                berger_files: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['berger']['bug'] == 1 else CommitLabel.NonBugFix),
                                herzig: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else CommitLabel.NonBugFix),
                                herzig_eval: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else CommitLabel.NonBugFix),
                                herzig_train: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else CommitLabel.NonBugFix),
                                mauczka_files: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['mauczka']['hl_corrective'] == 1 else CommitLabel.NonBugFix),
                                })

bugginess_herzig = Task(name='bugginess_herzig', author='hlib', description='bug or not (herzig)', top_artifact=Commit,
                 labels=[CommitLabel.NonBugFix, CommitLabel.BugFix],
                 training_dataset=herzig_train,
                 test_datasets={herzig_eval:lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else CommitLabel.NonBugFix)})


dataset_debugging = Experiment('dataset_debugging', bugginess,
                               heuristics_classifier=f'bugginess/fine_grained_changes_transformer_90.py:'
                                                     f'bugginess/buggless_if_one_file_markdown_ext.py:'
                                                     f'bugginess/buggless_if_doc_extensions.py:'
                                                            f'bugginess/fine_grained_changes_transformer_80.py:'
                                                            f'bugginess/fine_grained_changes_transformer_70.py:'
                                                            f'bugginess/filemetrics:'
                                                            f'bugginess/small_change.py:'
                                                            f'bugginess/large_change.py:'
                                                            f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                            f'bugginess/keywords/buggless_keywords_lookup_in_message.py'
                                                            f'@4ad6fd42a2cf8e609bd15afc4c7e4f43bce67dd8')


all_heuristics_with_issues = Experiment('all_heuristics_with_issues', bugginess,
                                                       heuristics_classifier=f'bugginess/fine_grained_changes_transformer_90.py:'
                                                                             f'bugginess/fine_grained_changes_transformer_80.py:'
                                                                             f'bugginess/fine_grained_changes_transformer_70.py:'
                                                                             f'bugginess/filemetrics:'
                                                                             f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                             f'bugginess/keywords/buggless_keywords_lookup_in_message.py:'
                                                                             f'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py:'
                                                                             f'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py:'
                                                                             f'bugginess/keywords/bug_keywords_lookup_in_issue_body.py:'
                                                                             f'bugginess/keywords/bug_keywords_lookup_in_issue_label.py'
                                                                             f'@4ad6fd42a2cf8e609bd15afc4c7e4f43bce67dd8')


all_heuristics_without_issues = Experiment('all_heuristics_without_issues', bugginess,
                                        heuristics_classifier=f'bugginess/fine_grained_changes_transformer_90.py:'
                                                              f'bugginess/fine_grained_changes_transformer_80.py:'
                                                              f'bugginess/fine_grained_changes_transformer_70.py:'
                                                              f'bugginess/filemetrics:'
                                                              f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                              f'bugginess/keywords/buggless_keywords_lookup_in_message.py'
                                                              f'@4ad6fd42a2cf8e609bd15afc4c7e4f43bce67dd8')


gitcproc = Experiment('gitcproc', bugginess, heuristics_classifier=f'bugginess/gitcproc@4ad6fd42a2cf8e609bd15afc4c7e4f43bce67dd8')

gitcproc_orig = Experiment('gitcproc_orig', bugginess, heuristics_classifier=f'bugginess/gitcproc/keywords.py@4ad6fd42a2cf8e609bd15afc4c7e4f43bce67dd8')

only_message_keywords = Experiment('only_message_keywords', bugginess, heuristics_classifier=f'bugginess/keywords/bug_keywords_lookup_in_message.py:bugginess/keywords/buggless_keywords_lookup_in_message.py@4ad6fd42a2cf8e609bd15afc4c7e4f43bce67dd8')
only_keywords = Experiment('only_keywords', bugginess, heuristics_classifier=f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                             f'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py:'
                                                                             f'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py:'
                                                                             f'bugginess/keywords/bug_keywords_lookup_in_issue_body.py:'
                                                                             f'bugginess/keywords/bug_keywords_lookup_in_issue_label.py:'
                                                                             f'bugginess/keywords/buggless_keywords_lookup_in_message.py@4ad6fd42a2cf8e609bd15afc4c7e4f43bce67dd8')


w = Workspace('0.5.0rc2', [
    dataset_debugging,
    all_heuristics_without_issues,
    all_heuristics_with_issues,
    gitcproc,
    gitcproc_orig,
    only_message_keywords,
    only_keywords,
])

