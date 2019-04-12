/*
database: ecoai
*/

SET NAMES utf8;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
--  Table structure for `observation_chem_site`
-- ----------------------------
DROP TABLE IF EXISTS `observation_chem_site`;
CREATE TABLE `observation_chem_site` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `site_id` int(11) DEFAULT NULL,
  `file_date` datetime DEFAULT NULL,
  `chem_type` varchar(16) DEFAULT NULL,
  `chem_unit` varchar(8) DEFAULT NULL,
  `observed_time` datetime DEFAULT NULL,
  `chem_value` decimal(10,4) DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC COMMENT='站点污染物观测表';
