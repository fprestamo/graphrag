# stages-evaluation

End-to-end evaluations of individual BT-GraphRAG pipeline stages.

Each subdirectory is one evaluation harness: a small corpus, the
extracted intermediate state for that corpus, manually authored ground
truth, and scripts that exercise the target stage(s) and score the
output against the truth.

| Folder      | Stages exercised                                                                 |
|-------------|----------------------------------------------------------------------------------|
| `cger_cgrr` | Cross-Graph Entity Resolution + Cross-Graph Relationship-type Resolution         |

See `cger_cgrr/README.md` for the usage pattern; future evaluations
(e.g. for ETCDR end-to-end, ICUS, the temporal extractor itself)
should follow the same layout:

    <eval-name>/
      data/
        corpus/         # raw .txt input documents
        extracted/      # intermediate state written by extract step
        ground_truth/   # manually authored expected output (JSON)
      extract.py        # corpus -> extracted state
      evaluate.py       # run stage(s) on extracted state, score vs ground truth
