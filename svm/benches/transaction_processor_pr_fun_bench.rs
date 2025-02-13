#[cfg(feature = "dev-context-only-utils")]
use qualifier_attr::{field_qualifiers, qualifiers};
use solana_svm::{bench_utilis::*, transaction_processor::{TransactionProcessingConfig, TransactionProcessingEnvironment}};
use {
    solana_client::rpc_client::RpcClient,
    solana_sdk::{
        account::{AccountSharedData, ReadableAccount},
        pubkey::Pubkey,
    },
    solana_svm::transaction_processing_callback::TransactionProcessingCallback,
    std::{collections::HashMap, sync::RwLock},
};
use {
    
        solana_svm::account_loader::{
            collect_rent_from_account,  validate_fee_payer,  AccountLoader,// load_transaction, AccountLoader
             CheckedTransactionDetails, LoadedTransaction, // AccountUsagePattern,
            TransactionCheckResult, TransactionLoadResult, ValidatedTransactionDetails,
        },
        solana_svm::account_overrides::AccountOverrides,
        solana_svm::message_processor::MessageProcessor,
        solana_svm::nonce_info::NonceInfo,
        solana_svm::program_loader::load_program_with_pubkey, // get_program_modification_slot
        solana_svm::rollback_accounts::RollbackAccounts,
        // solana_svm::transaction_account_state_info::TransactionAccountStateInfo,
        solana_svm::transaction_error_metrics::TransactionErrorMetrics,
        solana_svm::transaction_execution_result::{ExecutedTransaction, TransactionExecutionDetails},
        // solana_svm::transaction_processing_callback::TransactionProcessingCallback,
        solana_svm::transaction_processing_result::{ProcessedTransaction, TransactionProcessingResult},
    
    log::debug,
    percentage::Percentage,
    solana_account::{state_traits::StateMut,  PROGRAM_OWNERS}, // AccountSharedData, ReadableAccount,
    solana_bpf_loader_program::syscalls::{
        create_program_runtime_environment_v1, create_program_runtime_environment_v2,
    },
    solana_clock::{Epoch, Slot},
    solana_compute_budget::compute_budget::ComputeBudget,
    solana_compute_budget_instruction::instructions_processor::process_compute_budget_instructions,
    solana_feature_set::{
        enable_transaction_loading_failure_fees, remove_accounts_executable_flag_checks,
        remove_rounding_in_fee_calculation, FeatureSet,
    },
    solana_fee_structure::{FeeBudgetLimits, FeeStructure},
    solana_hash::Hash,
    solana_instruction::TRANSACTION_LEVEL_STACK_HEIGHT,
    solana_log_collector::LogCollector,
    solana_measure::{measure::Measure, measure_us},
    solana_message::compiled_instruction::CompiledInstruction,
    solana_nonce::{
        state::{DurableNonce, State as NonceState},
        versions::Versions as NonceVersions,
    },
    solana_program_runtime::{
        invoke_context::{EnvironmentConfig, InvokeContext},
        loaded_programs::{
            ForkGraph, ProgramCache, ProgramCacheEntry, ProgramCacheForTxBatch,
            ProgramCacheMatchCriteria, ProgramRuntimeEnvironment,
        },
        solana_sbpf::{
            program::{BuiltinProgram, FunctionRegistry},
            vm::Config as VmConfig,
        },
        sysvar_cache::SysvarCache,
    },
    // solana_pubkey::Pubkey,
    solana_sdk::{
        inner_instruction::{InnerInstruction, InnerInstructionsList},
        rent_collector::RentCollector,
    },
    solana_sdk_ids::{native_loader, system_program},
    solana_svm_rent_collector::svm_rent_collector::SVMRentCollector,
    solana_svm_transaction::{svm_message::SVMMessage, svm_transaction::SVMTransaction},
    solana_timings::{ExecuteTimingType, ExecuteTimings},
    solana_transaction_context::{ExecutionRecord, TransactionContext},
    solana_transaction_error::{TransactionError, TransactionResult},
    solana_type_overrides::sync::{atomic::Ordering, Arc, RwLockReadGuard},
    std::{
        collections::{hash_map::Entry, HashSet},
        fmt::{Debug, Formatter},
        rc::Rc,
        sync::Weak,
    },
};

// use std::sync::{Arc, RwLock};

#[allow(deprecated)]
    use solana_sysvar::fees::Fees;
    use {
        // super::*,
       
            solana_svm::account_loader::LoadedTransactionAccount, // ValidatedTransactionDetails
            // solana_svm::nonce_info::NonceInfo,
            // solana_svm::rollback_accounts::RollbackAccounts,
            solana_svm::transaction_processing_callback::AccountState,
        
        solana_account::{create_account_shared_data_for_test, WritableAccount},
        solana_clock::Clock,
        solana_compute_budget::compute_budget_limits::ComputeBudgetLimits,
        solana_compute_budget_interface::ComputeBudgetInstruction,
        solana_epoch_schedule::EpochSchedule,
        solana_fee_calculator::FeeCalculator,
        solana_fee_structure::FeeDetails, // FeeStructure
        // solana_hash::Hash,
        solana_keypair::Keypair,
        solana_message::{LegacyMessage, Message, MessageHeader, SanitizedMessage},
        solana_nonce as nonce,
        solana_program_runtime::loaded_programs::{BlockRelation, ProgramCacheEntryType},
        solana_rent::Rent,
        solana_rent_debits::RentDebits,
        solana_reserved_account_keys::ReservedAccountKeys,
        solana_sdk::rent_collector::RENT_EXEMPT_RENT_EPOCH, // RentCollector
        solana_sdk_ids::{bpf_loader, sysvar}, // system_program
        solana_signature::Signature,
        solana_transaction::{sanitized::SanitizedTransaction, Transaction},
        // solana_transaction_context::TransactionContext,
        // solana_transaction_error::TransactionError,
        test_case::test_case,
    };

use solana_program::feature;
// use solana_program_runtime::loaded_programs::ForkGraph;
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
        // solana_compute_budget::compute_budget::ComputeBudget,
        // solana_sdk::feature_set::FeatureSet,
        // solana_bpf_loader_program::syscalls::create_program_runtime_environment_v1,
    };









    fn new_unchecked_sanitized_message(message: Message) -> SanitizedMessage {
        SanitizedMessage::Legacy(LegacyMessage::new(
            message,
            &ReservedAccountKeys::empty_key_set(),
        ))
    }

    struct TestForkGraph {}

    impl ForkGraph for TestForkGraph {
        fn relationship(&self, _a: Slot, _b: Slot) -> BlockRelation {
            BlockRelation::Unknown
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

    // #[test_case(1; "Check results too small")]
    // #[test_case(3; "Check results too large")]
    // #[should_panic(expected = "Length of check_results does not match length of sanitized_txs")]
    fn test_check_results_txs_length_mismatch(check_results_len: usize) {
        let sanitized_message = new_unchecked_sanitized_message(Message {
            account_keys: vec![Pubkey::new_from_array([0; 32])],
            header: MessageHeader::default(),
            instructions: vec![CompiledInstruction {
                program_id_index: 0,
                accounts: vec![],
                data: vec![],
            }],
            recent_blockhash: Hash::default(),
        });

        // Transactions, length 2.
        let sanitized_txs = vec![
            SanitizedTransaction::new_for_tests(
                sanitized_message,
                vec![Signature::new_unique()],
                false,
            );
            2
        ];

        let check_results = vec![
            TransactionCheckResult::Ok(CheckedTransactionDetails {
                nonce: None,
                lamports_per_signature: 0
            });
            check_results_len
        ];

        let batch_processor = TransactionBatchProcessor::<TestForkGraph>::default();
        let callback = MockBankCallback::default();

        batch_processor.load_and_execute_sanitized_transactions(
            &callback,
            &sanitized_txs,
            check_results,
            &TransactionProcessingEnvironment::default(),
            &TransactionProcessingConfig::default(),
        );
    }





    fn bench_load_and_execute_sanitized_transactions<FG: ForkGraph>(c: &mut Criterion){
        let sanitized_message = new_unchecked_sanitized_message(Message {
            account_keys: vec![Pubkey::new_from_array([0; 32])],
            header: MessageHeader::default(),
            instructions: vec![CompiledInstruction {
                program_id_index: 0,
                accounts: vec![],
                data: vec![],
            }],
            recent_blockhash: Hash::default(),
        });  

        // Transactions, length 2.
        let sanitized_txs = vec![
            SanitizedTransaction::new_for_tests(
                sanitized_message,
                vec![Signature::new_unique()],
                false,
            );
            2
        ];


        let batch_processor = TransactionBatchProcessor::<TestForkGraph>::default();
        let callback = MockBankCallback::default();
    
        c.bench_function("bench_load_and_execute_sanitized_transactions", |b| {
            b.iter( || {

                let check_results = vec![
            TransactionCheckResult::Ok(CheckedTransactionDetails {
                nonce: None,
                lamports_per_signature: 0
            });
            2
            ];
                
                batch_processor.load_and_execute_sanitized_transactions(
                    &callback,
                    &sanitized_txs,
                    check_results,
                    &TransactionProcessingEnvironment::default(),
                    &TransactionProcessingConfig::default(),
                );
                
            })
        });
    }

criterion_group!(benches, bench_load_and_execute_sanitized_transactions<TestForkGraph>);
criterion_main!(benches);























//     // Dummy implementation of ForkGraph for testing purposes
// // #[derive(Default)]
// // struct DummyForkGraph;

// // impl ForkGraph for DummyForkGraph {
// //     fn relationship(&self, _a: solana_clock::Slot, _b: solana_clock::Slot) -> solana_program_runtime::loaded_programs::BlockRelation {
// //         solana_program_runtime::loaded_programs::BlockRelation::Unknown
// //     }
// // }


// // pub struct TestAccountLoader<'a> {
// //     cache: RwLock<HashMap<Pubkey, AccountSharedData>>,
// //     rpc_client: &'a RpcClient,
// // }

// // impl<'a> TestAccountLoader<'a> {
// //     pub fn new(rpc_client: &'a RpcClient) -> Self {
// //         Self {
// //             cache: RwLock::new(HashMap::new()),
// //             rpc_client,
// //         }
// //     }
// // }

// /// Implementation of the SVM API's `TransactionProcessingCallback` interface.
// ///
// /// The SVM API requires this plugin be provided to provide the SVM with the
// /// ability to load accounts.
// ///
// /// In the Agave validator, this implementation is Bank, powered by AccountsDB.
// impl TransactionProcessingCallback for TestAccountLoader<'_> {
//     fn get_account_shared_data(&self, pubkey: &Pubkey) -> Option<AccountSharedData> {
//         if let Some(account) = self.cache.read().unwrap().get(pubkey) {
//             return Some(account.clone());
//         }

//         let account: AccountSharedData = self.rpc_client.get_account(pubkey).ok()?.into();
//         self.cache.write().unwrap().insert(*pubkey, account.clone());

//         Some(account)
//     }

//     fn account_matches_owners(&self, account: &Pubkey, owners: &[Pubkey]) -> Option<usize> {
//         self.get_account_shared_data(account)
//             .and_then(|account| owners.iter().position(|key| account.owner().eq(key)))
//     }
// }

    
//     /// Implementation of the SVM API's `TransactionProcessingCallback` interface.
//     ///
//     /// The SVM API requires this plugin be provided to provide the SVM with the
//     /// ability to load accounts.
//     ///
//     /// In the Agave validator, this implementation is Bank, powered by AccountsDB.
    
    

// // #[bench]
// fn my_benchmark<FG: ForkGraph>(c: &mut Criterion){
//     // let fork_graph = RwLock::new(DummyForkGraph::default());
//     let processor = TransactionBatchProcessor::<DummyForkGraph>::new_uninitialized(1, 1);
//     let feature_set = FeatureSet::default();
//     let compute_budget = ComputeBudget::default();
//     let fork_graph = Arc::new(RwLock::new(DummyForkGraph::default()));


//     let account_loader = TestAccountLoader::new(&self.rpc_client);


//     c.bench_function("new_transaction_batch_processor", |b| {
//         b.iter(|| {
//             // Perform benchmarking logic
//             let processor = TransactionBatchProcessor::<DummyForkGraph>::new(
//                 /* slot */ 1,
//                 /* epoch */ 1,
//                 Arc::downgrade(&fork_graph),
//                 Some(Arc::new(
//                     create_program_runtime_environment_v1(&feature_set, &compute_budget, false, false)
//                         .unwrap(),
//                 )),
//                 None,
//             );


//             // let a = TransactionBatchProcessor::load_and_execute_sanitized_transactions(
//             //     &processor, 
//             //     callbacks, 
//             //     sanitized_txs, 
//             //     check_results, 
//             //     environment, 
//             //     config
//             // );
            
            
//         })
//     });
// }

// criterion_group!(benches, my_benchmark<DummyForkGraph>);
// criterion_main!(benches);




// #[derive(Default, Clone)]
//     pub struct MockBankCallback {
//         pub account_shared_data: Arc<RwLock<HashMap<Pubkey, AccountSharedData>>>,
//         #[allow(clippy::type_complexity)]
//         pub inspected_accounts:
//             Arc<RwLock<HashMap<Pubkey, Vec<(Option<AccountSharedData>, /* is_writable */ bool)>>>>,
//     }

//     impl TransactionProcessingCallback for MockBankCallback {
//         fn account_matches_owners(&self, account: &Pubkey, owners: &[Pubkey]) -> Option<usize> {
//             if let Some(data) = self.account_shared_data.read().unwrap().get(account) {
//                 if data.lamports() == 0 {
//                     None
//                 } else {
//                     owners.iter().position(|entry| data.owner() == entry)
//                 }
//             } else {
//                 None
//             }
//         }

//         fn get_account_shared_data(&self, pubkey: &Pubkey) -> Option<AccountSharedData> {
//             self.account_shared_data
//                 .read()
//                 .unwrap()
//                 .get(pubkey)
//                 .cloned()
//         }

//         fn add_builtin_account(&self, name: &str, program_id: &Pubkey) {
//             let mut account_data = AccountSharedData::default();
//             account_data.set_data(name.as_bytes().to_vec());
//             self.account_shared_data
//                 .write()
//                 .unwrap()
//                 .insert(*program_id, account_data);
//         }

//         fn inspect_account(
//             &self,
//             address: &Pubkey,
//             account_state: AccountState,
//             is_writable: bool,
//         ) {
//             let account = match account_state {
//                 AccountState::Dead => None,
//                 AccountState::Alive(account) => Some(account.clone()),
//             };
//             self.inspected_accounts
//                 .write()
//                 .unwrap()
//                 .entry(*address)
//                 .or_default()
//                 .push((account, is_writable));
//         }
//     }

    // impl<'a> From<&'a MockBankCallback> for AccountLoader<'a, MockBankCallback> {
    //     fn from(callbacks: &'a MockBankCallback) -> AccountLoader<'a, MockBankCallback> {
    //         AccountLoader::new_with_account_cache_capacity(
    //             None,
    //             ProgramCacheForTxBatch::default(),
    //             HashMap::default(),
    //             callbacks,
    //             Arc::<FeatureSet>::default(),
    //             0,
    //         )
    //     }
    // }
