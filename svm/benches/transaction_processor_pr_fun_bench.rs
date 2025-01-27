use solana_program_runtime::loaded_programs::ForkGraph;
// use solana_svm::transaction_processor_test::*;
use {
        criterion::{
            criterion_group, 
            criterion_main, 
            Criterion
        },
        solana_svm::
        {
            transaction_processor_pr_fun::TransactionBatchProcessor, 
            transaction_processor::TransactionBatchProcessor::new,
        }
    };

// Dummy implementation of ForkGraph for testing purposes
#[derive(Default)]
struct DummyForkGraph;

impl ForkGraph for DummyForkGraph {
    fn relationship(&self, _a: solana_clock::Slot, _b: solana_clock::Slot) -> solana_program_runtime::loaded_programs::BlockRelation {
        solana_program_runtime::loaded_programs::BlockRelation::Unknown
    }
}

// #[bench]
fn my_benchmark<FG: ForkGraph>(c: &mut Criterion){
    // let fork_graph = RwLock::new(DummyForkGraph::default());
    // let processor = TransactionBatchProcessor::<DummyForkGraph>::new_uninitialized(1, 1);

    c.bench_function("transaction_batch_processor", |b| {
        b.iter(|| {
            // Perform benchmarking logic
            let a = TransactionBatchProcessor::load_and_execute_sanitized_transactions(&self, callbacks, sanitized_txs, check_results, environment, config);
            
            
        })
    });
}

criterion_group!(benches, my_benchmark<DummyForkGraph>);
criterion_main!(benches);

