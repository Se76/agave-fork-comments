// use {
//     // core::ReplayStage::ReplayBlockstoreComponents,
//     crate::{
//         // consensus::{
//         //     progress_map::{ForkProgress, ValidatorStakeInfo, RETRANSMIT_BASE_DELAY_MS},
//         //     // tower_storage::{FileTowerStorage, NullTowerStorage},
//         //     tree_diff::TreeDiff,
//         //     ThresholdDecision, Tower, VOTE_THRESHOLD_DEPTH,
//         // },
//         // replay_stage::ReplayStage,
//         replay_stage::
//     {
//         // ReplayBlockstoreComponents,
//         tests::ReplayBlockstoreComponents, ReplayStage
//         // ReplayStage::ReplayBlockstoreComponents,

//     }, vote_simulator::{self, VoteSimulator}
//     },
//     // crossbeam_channel::unbounded,
//     // itertools::Itertools,
//     // solana_client::connection_cache::ConnectionCache,
//     // solana_entry::entry::{self, Entry},
//     // solana_gossip::{cluster_info::Node, crds::Cursor},
//     solana_ledger::blockstore::{entries_to_test_shreds, make_slot_entries, BlockstoreError},
//     // solana_rpc::{
//     //     optimistically_confirmed_bank_tracker::OptimisticallyConfirmedBank,
//     //     rpc::{create_test_transaction_entries, populate_blockstore_for_tests},
//     //     slot_status_notifier::SlotStatusNotifierInterface,
//     // },
//     solana_runtime::
//     // {
//         // accounts_background_service::AbsRequestSender,
//         // commitment::{BlockCommitment, VOTE_THRESHOLD_SIZE},
//         // genesis_utils::{GenesisConfigInfo, ValidatorVoteKeypairs},
//         bank::Bank,
//     // },
//     solana_sdk::
//     // {
//         clock::NUM_CONSECUTIVE_LEADER_SLOTS,
//         // genesis_config,
//         // hash::{hash, Hash},
//         // instruction::InstructionError,
//         // poh_config::PohConfig,
//         // signature::{Keypair, Signer},
//         // system_transaction,
//         // transaction::TransactionError,
//     // },
//     // solana_streamer::socket::SocketAddrSpace,
//     // solana_tpu_client::tpu_client::{DEFAULT_TPU_CONNECTION_POOL_SIZE, DEFAULT_VOTE_USE_QUIC},
//     // solana_transaction_status::VersionedTransactionWithStatusMeta,
//     // solana_vote_program::{
//     //     vote_state::{self, TowerSync, VoteStateVersions},
//     //     vote_transaction,
//     // },
//     // std::{
//     //     fs::remove_dir_all,
//     //     iter,
//     //     sync::{atomic::AtomicU64, Arc, Mutex, RwLock},
//     // },
//     // tempfile::tempdir,
//     // test_case::test_case,
//     // // trees::{tr, Tree},
//     // replay_stage::
//     // {
//     //     ReplayBlockstoreComponents,
//     //     ReplayStage::ReplayBlockstoreComponents,

//     // },
//     // replay_stage::ReplayStage,
// };

// // impl ReplayStage {
// //     fn generate_new_bank_forks(
// //         blockstore: &Blockstore,
// //         bank_forks: &RwLock<BankForks>,
// //         leader_schedule_cache: &Arc<LeaderScheduleCache>,
// //         rpc_subscriptions: &Arc<RpcSubscriptions>,
// //         slot_status_notifier: &Option<SlotStatusNotifier>,
// //         progress: &mut ProgressMap,
// //         replay_timing: &mut ReplayLoopTiming,
// //     ) {
// //         // Find the next slot that chains to the old slot
// //         let mut generate_new_bank_forks_read_lock =
// //             Measure::start("generate_new_bank_forks_read_lock");
// //         let forks = bank_forks.read().unwrap();
// //         generate_new_bank_forks_read_lock.stop();

// //         let frozen_banks = forks.frozen_banks();
// //         let frozen_bank_slots: Vec<u64> = frozen_banks
// //             .keys()
// //             .cloned()
// //             .filter(|s| *s >= forks.root())
// //             .collect();
// //         let mut generate_new_bank_forks_get_slots_since =
// //             Measure::start("generate_new_bank_forks_get_slots_since");
// //         let next_slots = blockstore
// //             .get_slots_since(&frozen_bank_slots)
// //             .expect("Db error");
// //         generate_new_bank_forks_get_slots_since.stop();

// //         // Filter out what we've already seen
// //         trace!("generate new forks {:?}", {
// //             let mut next_slots = next_slots.iter().collect::<Vec<_>>();
// //             next_slots.sort();
// //             next_slots
// //         });
// //         let mut generate_new_bank_forks_loop = Measure::start("generate_new_bank_forks_loop");
// //         let mut new_banks = HashMap::new();
// //         for (parent_slot, children) in next_slots {
// //             let parent_bank = frozen_banks
// //                 .get(&parent_slot)
// //                 .expect("missing parent in bank forks");
// //             for child_slot in children {
// //                 if forks.get(child_slot).is_some() || new_banks.contains_key(&child_slot) {
// //                     trace!("child already active or frozen {}", child_slot);
// //                     continue;
// //                 }
// //                 let leader = leader_schedule_cache
// //                     .slot_leader_at(child_slot, Some(parent_bank))
// //                     .unwrap();
// //                 info!(
// //                     "new fork:{} parent:{} root:{}",
// //                     child_slot,
// //                     parent_slot,
// //                     forks.root()
// //                 );
// //                 let child_bank = Self::new_bank_from_parent_with_notify(
// //                     parent_bank.clone(),
// //                     child_slot,
// //                     forks.root(),
// //                     &leader,
// //                     rpc_subscriptions,
// //                     slot_status_notifier,
// //                     NewBankOptions::default(),
// //                 );
// //                 let empty: Vec<Pubkey> = vec![];
// //                 Self::update_fork_propagated_threshold_from_votes(
// //                     progress,
// //                     empty,
// //                     vec![leader],
// //                     parent_bank.slot(),
// //                     &forks,
// //                 );
// //                 new_banks.insert(child_slot, child_bank);
// //             }
// //         }
// //         drop(forks);
// //         generate_new_bank_forks_loop.stop();

// //         let mut generate_new_bank_forks_write_lock =
// //             Measure::start("generate_new_bank_forks_write_lock");
// //         let mut forks = bank_forks.write().unwrap();
// //         for (_, bank) in new_banks {
// //             forks.insert(bank);
// //         }
// //         generate_new_bank_forks_write_lock.stop();
// //         saturating_add_assign!(
// //             replay_timing.generate_new_bank_forks_read_lock_us,
// //             generate_new_bank_forks_read_lock.as_us()
// //         );
// //         saturating_add_assign!(
// //             replay_timing.generate_new_bank_forks_get_slots_since_us,
// //             generate_new_bank_forks_get_slots_since.as_us()
// //         );
// //         saturating_add_assign!(
// //             replay_timing.generate_new_bank_forks_loop_us,
// //             generate_new_bank_forks_loop.as_us()
// //         );
// //         saturating_add_assign!(
// //             replay_timing.generate_new_bank_forks_write_lock_us,
// //             generate_new_bank_forks_write_lock.as_us()
// //         );
// //     }
// // }




// #[derive(Default)]
// struct ReplayLoopTiming {
//     last_submit: u64,
//     loop_count: u64,
//     collect_frozen_banks_elapsed_us: u64,
//     compute_bank_stats_elapsed_us: u64,
//     select_vote_and_reset_forks_elapsed_us: u64,
//     start_leader_elapsed_us: u64,
//     reset_bank_elapsed_us: u64,
//     voting_elapsed_us: u64,
//     generate_vote_us: u64,
//     update_commitment_cache_us: u64,
//     select_forks_elapsed_us: u64,
//     compute_slot_stats_elapsed_us: u64,
//     generate_new_bank_forks_elapsed_us: u64,
//     replay_active_banks_elapsed_us: u64,
//     wait_receive_elapsed_us: u64,
//     heaviest_fork_failures_elapsed_us: u64,
//     bank_count: u64,
//     process_ancestor_hashes_duplicate_slots_elapsed_us: u64,
//     process_duplicate_confirmed_slots_elapsed_us: u64,
//     process_duplicate_slots_elapsed_us: u64,
//     process_unfrozen_gossip_verified_vote_hashes_elapsed_us: u64,
//     process_popular_pruned_forks_elapsed_us: u64,
//     repair_correct_slots_elapsed_us: u64,
//     retransmit_not_propagated_elapsed_us: u64,
//     generate_new_bank_forks_read_lock_us: u64,
//     generate_new_bank_forks_get_slots_since_us: u64,
//     generate_new_bank_forks_loop_us: u64,
//     generate_new_bank_forks_write_lock_us: u64,
//     // When processing multiple forks concurrently, only captures the longest fork
//     replay_blockstore_us: u64,
// }

// // #[test]
//     fn test_child_slots_of_same_parent() {
//         let ReplayBlockstoreComponents {
//             blockstore,
//             validator_node_to_vote_keys,
//             vote_simulator,
//             leader_schedule_cache,
//             rpc_subscriptions,
//             ..
//         } = replay_blockstore_components(None, 1, None::<GenerateVotes>);

//         let VoteSimulator {
//             mut progress,
//             bank_forks,
//             ..
//         } = vote_simulator;

//         // Insert a non-root bank so that the propagation logic will update this
//         // bank
//         let bank1 = Bank::new_from_parent(
//             bank_forks.read().unwrap().get(0).unwrap(),
//             &leader_schedule_cache.slot_leader_at(1, None).unwrap(),
//             1,
//         );
//         progress.insert(
//             1,
//             ForkProgress::new_from_bank(
//                 &bank1,
//                 bank1.collector_id(),
//                 validator_node_to_vote_keys
//                     .get(bank1.collector_id())
//                     .unwrap(),
//                 Some(0),
//                 0,
//                 0,
//             ),
//         );
//         assert!(progress.get_propagated_stats(1).unwrap().is_leader_slot);
//         bank1.freeze();
//         bank_forks.write().unwrap().insert(bank1);

//         // Insert shreds for slot NUM_CONSECUTIVE_LEADER_SLOTS,
//         // chaining to slot 1
//         let (shreds, _) = make_slot_entries(
//             NUM_CONSECUTIVE_LEADER_SLOTS, // slot
//             1,                            // parent_slot
//             8,                            // num_entries
//             true,                         // merkle_variant
//         );
//         blockstore.insert_shreds(shreds, None, false).unwrap();
//         assert!(bank_forks
//             .read()
//             .unwrap()
//             .get(NUM_CONSECUTIVE_LEADER_SLOTS)
//             .is_none());
//         let mut replay_timing = ReplayLoopTiming::default();
//         ReplayStage::generate_new_bank_forks(
//             &blockstore,
//             &bank_forks,
//             &leader_schedule_cache,
//             &rpc_subscriptions,
//             &None,
//             &mut progress,
//             &mut replay_timing,
//         );
//         assert!(bank_forks
//             .read()
//             .unwrap()
//             .get(NUM_CONSECUTIVE_LEADER_SLOTS)
//             .is_some());

//         // Insert shreds for slot 2 * NUM_CONSECUTIVE_LEADER_SLOTS,
//         // chaining to slot 1
//         let (shreds, _) = make_slot_entries(
//             2 * NUM_CONSECUTIVE_LEADER_SLOTS,
//             1,
//             8,
//             true, // merkle_variant
//         );
//         blockstore.insert_shreds(shreds, None, false).unwrap();
//         assert!(bank_forks
//             .read()
//             .unwrap()
//             .get(2 * NUM_CONSECUTIVE_LEADER_SLOTS)
//             .is_none());
//         ReplayStage::generate_new_bank_forks(
//             &blockstore,
//             &bank_forks,
//             &leader_schedule_cache,
//             &rpc_subscriptions,
//             &None,
//             &mut progress,
//             &mut replay_timing,
//         );
//         assert!(bank_forks
//             .read()
//             .unwrap()
//             .get(NUM_CONSECUTIVE_LEADER_SLOTS)
//             .is_some());
//         assert!(bank_forks
//             .read()
//             .unwrap()
//             .get(2 * NUM_CONSECUTIVE_LEADER_SLOTS)
//             .is_some());

//         // // There are 20 equally staked accounts, of which 3 have built
//         // banks above or at bank 1. Because 3/20 < SUPERMINORITY_THRESHOLD,
//         // we should see 3 validators in bank 1's propagated_validator set.
//         let expected_leader_slots = vec![
//             1,
//             NUM_CONSECUTIVE_LEADER_SLOTS,
//             2 * NUM_CONSECUTIVE_LEADER_SLOTS,
//         ];
//         for slot in expected_leader_slots {
//             let leader = leader_schedule_cache.slot_leader_at(slot, None).unwrap();
//             let vote_key = validator_node_to_vote_keys.get(&leader).unwrap();
//             assert!(progress
//                 .get_propagated_stats(1)
//                 .unwrap()
//                 .propagated_validators
//                 .contains(vote_key));
//         }
//     }