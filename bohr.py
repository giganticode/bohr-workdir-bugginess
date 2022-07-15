from typing import Dict, Optional, List


from bohrapi.artifacts import Commit
from bohrlabels.core import LabelSet
from bohrlabels.labels import CommitLabel
from bohrapi.core import Dataset, Workspace, Experiment, LabelingTask


def id_with_files(id: str, conditions: Optional[List[Dict]] = None) -> Dict:
    return {"$and": [
        {id: {"$exists": True}},
        {"files": {"$exists": True}},
        {"files": {"$type": "array"}}] + (conditions if conditions is not None else [])}


berger_files = Dataset(id='berger_files', heuristic_input_artifact_type=Commit, query=id_with_files('manual_labels.berger'))
levin_files = Dataset(id='levin_files', heuristic_input_artifact_type=Commit, query=id_with_files('manual_labels.levin'))
herzig = Dataset(id='manual_labels.herzig', heuristic_input_artifact_type=Commit)
mauczka_files = Dataset(id='mauczka_files', heuristic_input_artifact_type=Commit, query=id_with_files('manual_labels.mauczka'))

REVISION = 'e885ed829a5ad95cef410836df081d5fb0081e6f'

# 74ddbae8a13ce1cfb468961e4cefba6f761f5cd9

bugginess = LabelingTask(name='bugginess', author='hlib', description='bug or not', heuristic_input_artifact_type=Commit,
                 labels=(CommitLabel.NonBugFix, CommitLabel.BugFix),
                 test_datasets={
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


refactoring_few_ref_heuristics = Experiment('refactoring_few_ref_heuristics', refactoring, train_dataset=levin_files,
                                           heuristics_classifier=f'@{REVISION}')


all_heuristics_with_issues = Experiment('all_heuristics_with_issues', bugginess,
                                        train_dataset=levin_files,
                                                       heuristics_classifier=f'@{REVISION}')


w = Workspace('0.7.0', [
    all_heuristics_with_issues,
    refactoring_few_ref_heuristics,
])

