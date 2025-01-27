use std::sync::{Arc, RwLock};


use solana_program::feature;
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
            // transaction_processor::TransactionBatchProcessor::new,
        },
        solana_compute_budget::compute_budget::ComputeBudget,
        solana_sdk::feature_set::FeatureSet,
        solana_bpf_loader_program::syscalls::create_program_runtime_environment_v1,
    };

// Dummy implementation of ForkGraph for testing purposes
#[derive(Default)]
struct DummyForkGraph;

impl ForkGraph for DummyForkGraph {
    fn relationship(&self, _a: solana_clock::Slot, _b: solana_clock::Slot) -> solana_program_runtime::loaded_programs::BlockRelation {
        solana_program_runtime::loaded_programs::BlockRelation::Unknown
    }
}

#[derive(Default, Clone)]
    pub struct MockBankCallback {
        pub account_shared_data: Arc<RwLock<HashMap<Pubkey, AccountSharedData>>>,
        #[allow(clippy::type_complexity)]
        pub inspected_accounts:
            Arc<RwLock<HashMap<Pubkey, Vec<(Option<AccountSharedData>, /* is_writable */ bool)>>>>,
    }

    impl TransactionProcessingCallback for MockBankCallback {
        fn account_matches_owners(&self, account: &Pubkey, owners: &[Pubkey]) -> Option<usize> {
            if let Some(data) = self.account_shared_data.read().unwrap().get(account) {
                if data.lamports() == 0 {
                    None
                } else {
                    owners.iter().position(|entry| data.owner() == entry)
                }
            } else {
                None
            }
        }

        fn get_account_shared_data(&self, pubkey: &Pubkey) -> Option<AccountSharedData> {
            self.account_shared_data
                .read()
                .unwrap()
                .get(pubkey)
                .cloned()
        }

        fn add_builtin_account(&self, name: &str, program_id: &Pubkey) {
            let mut account_data = AccountSharedData::default();
            account_data.set_data(name.as_bytes().to_vec());
            self.account_shared_data
                .write()
                .unwrap()
                .insert(*program_id, account_data);
        }

        fn inspect_account(
            &self,
            address: &Pubkey,
            account_state: AccountState,
            is_writable: bool,
        ) {
            let account = match account_state {
                AccountState::Dead => None,
                AccountState::Alive(account) => Some(account.clone()),
            };
            self.inspected_accounts
                .write()
                .unwrap()
                .entry(*address)
                .or_default()
                .push((account, is_writable));
        }
    }

    impl<'a> From<&'a MockBankCallback> for AccountLoader<'a, MockBankCallback> {
        fn from(callbacks: &'a MockBankCallback) -> AccountLoader<'a, MockBankCallback> {
            AccountLoader::new_with_account_cache_capacity(
                None,
                ProgramCacheForTxBatch::default(),
                HashMap::default(),
                callbacks,
                Arc::<FeatureSet>::default(),
                0,
            )
        }
    }

// #[bench]
fn my_benchmark<FG: ForkGraph>(c: &mut Criterion){
    // let fork_graph = RwLock::new(DummyForkGraph::default());
    let processor = TransactionBatchProcessor::<DummyForkGraph>::new_uninitialized(1, 1);
    let feature_set = FeatureSet::default();
    let compute_budget = ComputeBudget::default();
    let fork_graph = Arc::new(RwLock::new(DummyForkGraph::default()));

    c.bench_function("new_transaction_batch_processor", |b| {
        b.iter(|| {
            // Perform benchmarking logic
            let processor = TransactionBatchProcessor::<DummyForkGraph>::new(
                /* slot */ 1,
                /* epoch */ 1,
                Arc::downgrade(&fork_graph),
                Some(Arc::new(
                    create_program_runtime_environment_v1(&feature_set, &compute_budget, false, false)
                        .unwrap(),
                )),
                None,
            );


            // let a = TransactionBatchProcessor::load_and_execute_sanitized_transactions(
            //     &processor, 
            //     callbacks, 
            //     sanitized_txs, 
            //     check_results, 
            //     environment, 
            //     config
            // );
            
            
        })
    });
}

criterion_group!(benches, my_benchmark<DummyForkGraph>);
criterion_main!(benches);

