USE `asign_system`;
/* 文件表增加打印分配次数字段 */
ALTER TABLE `asign_system`.`sign_sys_file_resource` ADD COLUMN `print_dis_num` BIGINT(20) NULL COMMENT '最后一次打印分配次数' AFTER `gmt_mod`;
