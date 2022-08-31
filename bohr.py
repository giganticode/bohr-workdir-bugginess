from typing import Dict, Optional, List, Any

from pydot import frozendict

from bohrapi.artifacts import Commit
from bohrlabels.core import LabelSet
from bohrlabels.labels import CommitLabel
from bohrapi.core import Dataset, Workspace, Experiment, LabelingTask


def id_with_files(id: str, conditions: Optional[List[Dict]] = None) -> Dict:
    return {"$and": [
        {id: {"$exists": True}},
        {"files": {"$exists": True}},
        {"files": {"$type": "array"}}] + (conditions if conditions is not None else [])}


important_projects= {
    'levin' : ["apache/hadoop", "apache/hbase", "ReactiveX/RxJava", "apache/camel", "elastic/elasticsearch", "JetBrains/intellij-community", "restlet/restlet-framework-java", "spring-projects/spring-framework", "kiegroup/drools", "orientechnologies/orientdb", "JetBrains/kotlin"],
    'berger': ["0x43/DesignPatternsPHP", "AlexMeliq/less.js", "Arcank/nimbus", "AutoMapper/AutoMapper", "Chenkaiang/XVim", "GeertJohan/gorp", "K2InformaticsGmBH/proper", "MerlinDMC/gocode", "MythTV/mythtv", "alibaba/tengine", "clojure/core.logic", "docpad/docpad", "faylang/fay", "lfe/lfe", "magicalpanda/MagicalRecord", "mpeltonen/sbt-idea", "plumatic/plumbing", "sinclairzx81/typescript.api", "yu19930123/ngrok", "JetBrains/kotlin"],
    'herzig': ["apache/jackrabbit", "apache/lucene-solr", "apache/tomcat", "mozilla/rhino", "apache/httpcomponents-client"],
}


def important_project_query(datasets: List[str]) -> Dict[str, Any]:
    projects = [p for dataset in datasets for p in important_projects[dataset]]

    important_project_condition = []
    for project in projects:
        owner, repo = project.split('/')
        important_project_condition.append({'owner': owner, 'repo': repo})

    query = {'$and': [{'files': {"$exists": True}}, {"files": {"$type": "array"}}, {"message": {"$exists": True}}, {"special_commit_finder/0_1.merge": False}, {"manual_labels": {"$exists": False}}, {'$or': important_project_condition}]}
    return query


commits_200k = Dataset(id='commits_200k', heuristic_input_artifact_type=Commit, query={'bohr.200k_commits': {"$exists": True}})
commits_200k_files = Dataset(id='commits_200k_files', heuristic_input_artifact_type=Commit, query=id_with_files('bohr.200k_commits'))
commits_200k_files_0 = Dataset(id='commits_200k_files_0', heuristic_input_artifact_type=Commit, query=id_with_files('bohr.200k_commits', [{"_id": {'$regex':'^0'}}]))
commits_200k_files_00 = Dataset(id='commits_200k_files_00', heuristic_input_artifact_type=Commit, query=id_with_files('bohr.200k_commits', [{"_id": {'$regex':'^00'}}]))
commits_200k_files_000 = Dataset(id='commits_200k_files_000', heuristic_input_artifact_type=Commit, query=id_with_files('bohr.200k_commits', [{"_id": {'$regex':'^000'}}]))
commits_200k_files_no_merges = Dataset(id='commits_200k_files_no_merges', heuristic_input_artifact_type=Commit, query=id_with_files("bohr.200k_commits", [{"special_commit_finder/0_1.merge": False}]))
levin_berger_herzig_train = Dataset(id='levin_berger_herzig_train', heuristic_input_artifact_type=Commit, query=important_project_query(['levin', 'berger', 'herzig']), projection={'gumtree/3_0_0-beta2': 0}, n_datapoints=20000)
levin_train = Dataset(id='levin_train', heuristic_input_artifact_type=Commit, query=important_project_query(['levin']), projection={'gumtree/3_0_0-beta2': 0}, n_datapoints=20000)
berger_train = Dataset(id='berger_train', heuristic_input_artifact_type=Commit, query=important_project_query(['berger']), projection={'gumtree/3_0_0-beta2': 0}, n_datapoints=20000)
herzig_train = Dataset(id='herzig_train', heuristic_input_artifact_type=Commit, query=important_project_query(['herzig']), projection={'gumtree/3_0_0-beta2': 0}, n_datapoints=20000)

berger_files = Dataset(id='berger_files', heuristic_input_artifact_type=Commit, query=id_with_files('manual_labels.berger'))
levin_files = Dataset(id='levin_files', heuristic_input_artifact_type=Commit, query=id_with_files('manual_labels.levin'))
herzig = Dataset(id='manual_labels.herzig', heuristic_input_artifact_type=Commit)
mauczka_files = Dataset(id='mauczka_files', heuristic_input_artifact_type=Commit, query=id_with_files('manual_labels.mauczka'))
idan_files = Dataset(id='idan_files', heuristic_input_artifact_type=Commit, query=id_with_files('idan/0_1', [{"idan/0_1.Is_Corrective": {"$exists": True}}]))

bohr_herzig_train = Dataset(id='bohr.herzig_train', heuristic_input_artifact_type=Commit)
bohr_herzig_eval = Dataset(id='bohr.herzig_eval', heuristic_input_artifact_type=Commit)

levin_small_changes = Dataset(id='levin_small_changes', heuristic_input_artifact_type=Commit, query=id_with_files('manual_labels.levin', [{'bohr.gt_512_codeberta_tokens': False}]))
levin_large_changes = Dataset(id='levin_large_changes', heuristic_input_artifact_type=Commit, query=id_with_files('manual_labels.levin', [{'bohr.gt_512_codeberta_tokens': True}]))

berger_small_changes = Dataset(id='berger_small_changes', heuristic_input_artifact_type=Commit, query=id_with_files('manual_labels.berger', [{'bohr.gt_512_codeberta_tokens': False}]))
berger_large_changes = Dataset(id='berger_large_changes', heuristic_input_artifact_type=Commit, query=id_with_files('manual_labels.berger', [{'bohr.gt_512_codeberta_tokens': True}]))

bohr_200k_small_changes = Dataset(id='bohr_200k_small_changes', heuristic_input_artifact_type=Commit, query=id_with_files('bohr.200k_commits', [{'bohr.gt_512_codeberta_tokens': False}]))
bohr_200k_large_changes = Dataset(id='bohr_200k_large_changes', heuristic_input_artifact_type=Commit, query=id_with_files('bohr.200k_commits', [{'bohr.gt_512_codeberta_tokens': True}]))

#conventional_with_files = Dataset(id='conventional_with_files', top_artifact=Commit, query={'conventional_commit/0_1.conventional': True, 'message': {"$exists": True}, 'files': {"$exists": True}}, n_datapoints=20000)


bugginess = LabelingTask(name='bugginess', author='hlib', description='bug or not', heuristic_input_artifact_type=Commit,
                 labels=(CommitLabel.NonBugFix, CommitLabel.BugFix),
                 test_datasets={
                                idan_files: lambda c: (CommitLabel.BugFix if c.raw_data['idan/0_1']['Is_Corrective'] else CommitLabel.NonBugFix),
                                levin_files: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['levin']['bug'] == 1 else CommitLabel.NonBugFix),
                                berger_files: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['berger']['bug'] == 1 else CommitLabel.NonBugFix),
                                herzig: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else CommitLabel.NonBugFix),
                                mauczka_files: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['mauczka']['hl_corrective'] == 1 else CommitLabel.NonBugFix),
                                })

refactoring = LabelingTask(name='refactoring',
                           author='hlib',
                           description='refactoring or not',
                           heuristic_input_artifact_type=Commit,
                           labels=(~LabelSet.of(CommitLabel.Refactoring), LabelSet.of(CommitLabel.Refactoring)),
                           test_datasets={
                               herzig: lambda c: (CommitLabel.Refactoring if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'REFACTORING' else CommitLabel.CommitLabel & ~CommitLabel.Refactoring),
                           })

REVISION = '5e101b0d8dc87680d6f6b7af012feffbc55834be'

refactoring_no_ref_heuristics = Experiment('refactoring_no_ref_heuristics', refactoring, train_dataset=herzig_train,
                                      heuristics_classifier=f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                            f'bugginess/keywords/buggless_keywords_lookup_in_message.py'
                                                            f'@{REVISION}')

refactoring_few_ref_heuristics = Experiment('refactoring_few_ref_heuristics', refactoring, train_dataset=herzig_train,
                                           heuristics_classifier=f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                 f'bugginess/keywords/buggless_keywords_lookup_in_message.py:'
                                                                 f'refactoring/keywords.py'
                                                                 f'@{REVISION}')

dataset_debugging = Experiment('dataset_debugging', bugginess,
                               train_dataset=commits_200k_files_000,
                               extra_test_datasets=frozendict(
                                   {bohr_herzig_eval: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else CommitLabel.NonBugFix),
                                    bohr_herzig_train: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else CommitLabel.NonBugFix)}
                                ),
                               heuristics_classifier=f'refactoring/keywords.py:'
                                                     f'bugginess/fine_grained_changes_transformer_90.py:'
                                                     f'bugginess/buggless_if_one_file_markdown_ext.py:'
                                                     f'bugginess/buggless_if_doc_extensions.py:'
                                                            f'bugginess/fine_grained_changes_transformer_80.py:'
                                                            f'bugginess/fine_grained_changes_transformer_70.py:'
                                                            f'bugginess/filemetrics:'
                                                            f'bugginess/small_change.py:'
                                                            f'bugginess/large_change.py:'
                                                     f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                     f'bugginess/keywords/buggless_keywords_lookup_in_message.py:'
                                                     f'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py:'
                                                     f'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py:'
                                                     f'bugginess/keywords/bug_keywords_lookup_in_issue_body.py:'
                                                     f'bugginess/keywords/bug_keywords_lookup_in_issue_label.py'
                                                            f'@{REVISION}')


all_heuristics_with_issues = Experiment('all_heuristics_with_issues', bugginess,
                                        train_dataset=commits_200k_files,
                                                       heuristics_classifier=f'refactoring/keywords.py:'
                                                                             f'bugginess/fine_grained_changes_transformer_90.py:'
                                                                             f'bugginess/fine_grained_changes_transformer_80.py:'
                                                                             f'bugginess/fine_grained_changes_transformer_70.py:'
                                                                             f'bugginess/filemetrics:'
                                                                             f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                             f'bugginess/keywords/buggless_keywords_lookup_in_message.py:'
                                                                             f'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py:'
                                                                             f'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py:'
                                                                             f'bugginess/keywords/bug_keywords_lookup_in_issue_body.py:'
                                                                             f'bugginess/keywords/bug_keywords_lookup_in_issue_label.py'
                                                                             f'@{REVISION}')


all_heuristics_without_issues = Experiment('all_heuristics_without_issues', bugginess,
                                           train_dataset=commits_200k_files,
                                        heuristics_classifier=f'refactoring/keywords.py:'
                                                              f'bugginess/fine_grained_changes_transformer_90.py:'
                                                              f'bugginess/fine_grained_changes_transformer_80.py:'
                                                              f'bugginess/fine_grained_changes_transformer_70.py:'
                                                              f'bugginess/filemetrics:'
                                                              f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                              f'bugginess/keywords/buggless_keywords_lookup_in_message.py'
                                                              f'@{REVISION}')

all_heuristics_without_issues_orig200k = Experiment('all_heuristics_without_issues_orig200k', bugginess,
                                                 train_dataset=commits_200k,
                                           heuristics_classifier=f'refactoring/keywords.py:'
                                                                 f'bugginess/fine_grained_changes_transformer_90.py:'
                                                                 f'bugginess/fine_grained_changes_transformer_80.py:'
                                                                 f'bugginess/fine_grained_changes_transformer_70.py:'
                                                                 f'bugginess/filemetrics:'
                                                                 f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                 f'bugginess/keywords/buggless_keywords_lookup_in_message.py'
                                                                 f'@{REVISION}')


message_keywords_filemetrics = Experiment('message_keywords_filemetrics', bugginess,
                                          train_dataset=commits_200k_files,
                                           heuristics_classifier=f'refactoring/keywords.py:'
                                                                 f'bugginess/filemetrics:'
                                                                 f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                 f'bugginess/keywords/buggless_keywords_lookup_in_message.py'
                                                                 f'@{REVISION}')


gitcproc = Experiment('gitcproc', bugginess, train_dataset=commits_200k_files, heuristics_classifier=f'bugginess/gitcproc@{REVISION}')

gitcproc_orig = Experiment('gitcproc_orig', bugginess, train_dataset=commits_200k_files, heuristics_classifier=f'bugginess/gitcproc/keywords.py@{REVISION}')

only_message_keywords = Experiment('only_message_keywords', bugginess, train_dataset=commits_200k_files, heuristics_classifier=f'refactoring/keywords.py:'
                                                                                                                               f'bugginess/keywords/bug_keywords_lookup_in_message.py:bugginess/keywords/buggless_keywords_lookup_in_message.py@{REVISION}')
only_keywords = Experiment('only_keywords', bugginess, train_dataset=commits_200k_files, heuristics_classifier=f'refactoring/keywords.py:'
                                                                                                               f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                             f'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py:'
                                                                             f'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py:'
                                                                             f'bugginess/keywords/bug_keywords_lookup_in_issue_body.py:'
                                                                             f'bugginess/keywords/bug_keywords_lookup_in_issue_label.py:'
                                                                             f'bugginess/keywords/buggless_keywords_lookup_in_message.py@{REVISION}')

only_message_and_label_keywords = Experiment('only_message_and_label_keywords', bugginess, train_dataset=commits_200k_files, heuristics_classifier=f'refactoring/keywords.py:'
                                                                                                                                                   f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                                                               f'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py:'
                                                                                                               f'bugginess/keywords/bug_keywords_lookup_in_issue_label.py:'
                                                                                                               f'bugginess/keywords/buggless_keywords_lookup_in_message.py@{REVISION}')


only_message_keywords_important_projects = Experiment('only_message_keywords_important_projects', bugginess, train_dataset=levin_berger_herzig_train, heuristics_classifier=f'refactoring/keywords.py:'
                                                                                                                                                                            f'bugginess/keywords/bug_keywords_lookup_in_message.py:bugginess/keywords/buggless_keywords_lookup_in_message.py@{REVISION}')

all_heuristics_without_issues_important_projects = Experiment('all_heuristics_without_issues_important_projects', bugginess,
                                           train_dataset=levin_berger_herzig_train,
                                           heuristics_classifier=f'refactoring/keywords.py:'
                                                                 f'bugginess/fine_grained_changes_transformer_90.py:'
                                                                 f'bugginess/fine_grained_changes_transformer_80.py:'
                                                                 f'bugginess/fine_grained_changes_transformer_70.py:'
                                                                 f'bugginess/filemetrics:'
                                                                 f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                 f'bugginess/keywords/buggless_keywords_lookup_in_message.py'
                                                                 f'@{REVISION}')

only_message_keywords_levin = Experiment('only_message_keywords_levin', bugginess, train_dataset=levin_train, heuristics_classifier=f'refactoring/keywords.py:'
                                                                                                                                    f'bugginess/keywords/bug_keywords_lookup_in_message.py:bugginess/keywords/buggless_keywords_lookup_in_message.py@{REVISION}')

all_heuristics_without_issues_levin = Experiment('all_heuristics_without_issues_levin', bugginess,
                                                              train_dataset=levin_train,
                                                              heuristics_classifier=f'refactoring/keywords.py:'
                                                                                    f'bugginess/fine_grained_changes_transformer_90.py:'
                                                                                    f'bugginess/fine_grained_changes_transformer_80.py:'
                                                                                    f'bugginess/fine_grained_changes_transformer_70.py:'
                                                                                    f'bugginess/filemetrics:'
                                                                                    f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                                    f'bugginess/keywords/buggless_keywords_lookup_in_message.py'
                                                                                    f'@{REVISION}')

only_message_keywords_berger = Experiment('only_message_keywords_berger', bugginess, train_dataset=berger_train, heuristics_classifier=f'refactoring/keywords.py:'
                                                                                                                                       f'bugginess/keywords/bug_keywords_lookup_in_message.py:bugginess/keywords/buggless_keywords_lookup_in_message.py@{REVISION}')

all_heuristics_without_issues_berger = Experiment('all_heuristics_without_issues_berger', bugginess,
                                                 train_dataset=berger_train,
                                                 heuristics_classifier=f'refactoring/keywords.py:'
                                                                       f'bugginess/fine_grained_changes_transformer_90.py:'
                                                                       f'bugginess/fine_grained_changes_transformer_80.py:'
                                                                       f'bugginess/fine_grained_changes_transformer_70.py:'
                                                                       f'bugginess/filemetrics:'
                                                                       f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                       f'bugginess/keywords/buggless_keywords_lookup_in_message.py'
                                                                       f'@{REVISION}')


only_message_keywords_herzig = Experiment('only_message_keywords_herzig', bugginess, train_dataset=herzig_train, heuristics_classifier=f'refactoring/keywords.py:'
                                                                                                                                       f'bugginess/keywords/bug_keywords_lookup_in_message.py:bugginess/keywords/buggless_keywords_lookup_in_message.py@{REVISION}')

all_heuristics_without_issues_herzig = Experiment('all_heuristics_without_issues_herzig', bugginess,
                                                  train_dataset=herzig_train,
                                                  heuristics_classifier=f'refactoring/keywords.py:'
                                                                        f'bugginess/fine_grained_changes_transformer_90.py:'
                                                                        f'bugginess/fine_grained_changes_transformer_80.py:'
                                                                        f'bugginess/fine_grained_changes_transformer_70.py:'
                                                                        f'bugginess/filemetrics:'
                                                                        f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                        f'bugginess/keywords/buggless_keywords_lookup_in_message.py'
                                                                        f'@{REVISION}')

w = Workspace('0.7.0', [
    dataset_debugging,
    all_heuristics_without_issues,
    all_heuristics_without_issues_orig200k,
    all_heuristics_with_issues,
    # message_keywords_filemetrics,
    gitcproc,
    gitcproc_orig,
    only_message_keywords,
    only_message_and_label_keywords,
    only_keywords,
    only_message_keywords_important_projects,
    all_heuristics_without_issues_important_projects,
    all_heuristics_without_issues_levin,
    only_message_keywords_levin,
    all_heuristics_without_issues_berger,
    only_message_keywords_berger,
    all_heuristics_without_issues_herzig,
    only_message_keywords_herzig,
    refactoring_no_ref_heuristics,
    refactoring_few_ref_heuristics,
])

