from bohrapi.artifacts import Commit
from bohrlabels.labels import CommitLabel
from bohrapi.core import Dataset, Task, Workspace, Experiment

commits_200k = Dataset(id='bohr.200k_commits', top_artifact=Commit,)

berger = Dataset(id='manual_labels.berger', top_artifact=Commit)

levin = Dataset(id='manual_labels.levin', top_artifact=Commit)

herzig = Dataset(id='manual_labels.herzig', top_artifact=Commit)

bugginess = Task(name='bugginess', author='hlib', description='bug or not', top_artifact=Commit,
                 labels=[CommitLabel.NonBugFix, CommitLabel.BugFix],
                 training_dataset=commits_200k,
                 test_datasets={levin: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['levin']['bug'] == 1 else CommitLabel.NonBugFix),
                                berger: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['berger']['bug'] == 1 else CommitLabel.NonBugFix),
                                herzig: lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else CommitLabel.NonBugFix)})

exp = Experiment('only_keywords', bugginess,
                 heuristics_classifier=f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                       f'bugginess/keywords/buggless_keywords_lookup_in_message.py:'
                                       f'bugginess/keywords/bug_keywords_lookup_in_issue_body.py:'
                                       f'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py:'
                                       f'bugginess/keywords/bug_keywords_lookup_in_issue_label.py:'
                                       f'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py'
                                       f'@dddbe7ba63a14c718d08e7c88b166f90980fec05')

exp2 = Experiment('keywords_and_file_metrics', bugginess,
                 heuristics_classifier=f'bugginess/filemetrics:'
                                       f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                       f'bugginess/keywords/buggless_keywords_lookup_in_message.py:'
                                       f'bugginess/keywords/bug_keywords_lookup_in_issue_body.py:'
                                       f'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py:'
                                       f'bugginess/keywords/bug_keywords_lookup_in_issue_label.py:'
                                       f'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py'
                                       f'@dddbe7ba63a14c718d08e7c88b166f90980fec05')

filemetrics_and_transformer = Experiment('filemetrics_and_transformer', bugginess,
                              heuristics_classifier=f'bugginess/filemetrics:bugginess/fine_grained_changes_transformer.py@dddbe7ba63a14c718d08e7c88b166f90980fec05')

only_filemetrics = Experiment('only_filemetrics', bugginess,
                              heuristics_classifier=f'bugginess/filemetrics@dddbe7ba63a14c718d08e7c88b166f90980fec05')

keywords_filemetrics_transformer = Experiment('keywords_filemetrics_transformer', bugginess,
                                                     heuristics_classifier=f'bugginess/fine_grained_changes_transformer.py:'
                                                                           f'bugginess/filemetrics:'
                                                                           f'bugginess/keywords/bug_keywords_lookup_in_message.py:'
                                                                           f'bugginess/keywords/buggless_keywords_lookup_in_message.py:'
                                                                           f'bugginess/keywords/bug_keywords_lookup_in_issue_body.py:'
                                                                           f'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py:'
                                                                           f'bugginess/keywords/bug_keywords_lookup_in_issue_label.py:'
                                                                           f'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py'
                                                                           f'@dddbe7ba63a14c718d08e7c88b166f90980fec05')

w = Workspace('0.5.0rc0', [exp, exp2, filemetrics_and_transformer, only_filemetrics, keywords_filemetrics_transformer])

